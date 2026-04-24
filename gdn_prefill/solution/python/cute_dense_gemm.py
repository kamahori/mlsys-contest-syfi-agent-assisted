"""
From-scratch Blackwell SM100 dense GEMM in CuTe DSL.

A simple warp-specialized persistent TMA + tcgen05 BF16 GEMM.

  C[M,N] = A[M,K] @ B[N,K]^T       (A row-major, B row-major → C row-major)
                                     (K contiguous for both A and B)

Architecture:
  - 6 warps (192 threads): warps 0-3 = epilogue, warp 4 = MMA, warp 5 = TMA
  - Mainloop: TMA-load A/B tiles into SMEM → tcgen05 UMMA from SMEM → tmem
  - Epilogue: tmem → rmem → type convert → smem → TMA-store to gmem
  - Persistent tile scheduling across M×N output tiles.

Fixed config for simplicity:
  - mma_tiler_mn = (128, 128), cluster = (1, 1), no 2CTA
  - ab_dtype = BFloat16, acc_dtype = Float32, c_dtype = BFloat16
  - A: row-major (K contiguous), B: row-major (K contiguous), C: row-major (N contiguous)

References:
  - NVIDIA cutlass/examples/blackwell grouped_gemm (vendored in torch._inductor)
  - flashinfer gemm_allreduce_two_shot.py (PersistentDenseGemmKernel)
"""

from __future__ import annotations

import functools
from inspect import isclass
from typing import Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import torch


# ===========================================================================
# Kernel class
# ===========================================================================

class SimpleDenseGemm:
    """A from-scratch Blackwell sm100 BF16 dense GEMM kernel."""

    # Constants
    bytes_per_tensormap = 128
    num_tensormaps = 3  # A, B, C

    def __init__(self):
        # Fixed config
        mma_tiler_mn = (128, 128)
        cluster_shape_mn = (1, 1)
        use_2cta = False
        acc_dtype = cutlass.Float32

        self.acc_dtype = acc_dtype
        self.use_2cta = use_2cta
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)  # K dim set in _setup_attributes
        self.cta_group = tcgen05.CtaGroup.ONE

        # Warp roles (6 warps total = 192 threads)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192

        # Named barriers
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=32 * len(self.epilog_warp_id)
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * (len(self.epilog_warp_id) + 1)
        )

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0
        self.occupancy = 1
        self.buffer_align_bytes = 1024

        # No multicast for cluster=(1,1)
        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

    def _setup_attributes(self):
        """Compute MMA-dependent layouts and pipeline stages."""
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.acc_dtype, self.cta_group, self.mma_tiler[:2],
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0], self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = self.cta_tile_shape_mnk  # cluster=(1,1)

        # Cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Epilogue tile
        self.epi_tile = utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta, self.c_layout, self.c_dtype,
        )

        # Pipeline stages
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_epi_stage,
        ) = self._compute_stages(tiled_mma)

        # SMEM layouts
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_epi_stage,
        )

        # Tmem columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    def _compute_stages(self, tiled_mma):
        """Determine the number of pipeline stages for AB, ACC, and epilogue."""
        num_acc_stage = 2

        a_smem_layout_1 = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, 1,
        )
        b_smem_layout_1 = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, 1,
        )
        epi_smem_layout_1 = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1,
        )

        a_size = cute.size_in_bytes(self.a_dtype, a_smem_layout_1.outer)
        b_size = cute.size_in_bytes(self.b_dtype, b_smem_layout_1.outer)
        epi_size = cute.size_in_bytes(self.c_dtype, epi_smem_layout_1.outer)

        reserved_smem = 1024  # barriers + misc
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        ab_per_stage = (a_size + b_size) * atom_thr_size
        num_epi_stage = 2

        # Fill remaining SMEM with AB stages
        avail = self.smem_capacity * self.occupancy - reserved_smem - epi_size * num_epi_stage
        num_ab_stage = max(2, avail // ab_per_stage)
        # Cap to a reasonable limit
        num_ab_stage = min(num_ab_stage, 8)

        return num_acc_stage, num_ab_stage, num_epi_stage

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler, num_acc_stage):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    # -------------------------------------------------------------------
    # Epilogue partition helpers (from vendored kernel, handles layouts)
    # -------------------------------------------------------------------
    @cute.jit
    def epilog_tmem_copy_and_partition(self, tidx, tAcc, gC_mnl, epi_tile, use_2cta):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, self.c_layout, self.c_dtype,
            self.acc_dtype, epi_tile, use_2cta,
        )
        # tAcc is (MMA, MMA_M, MMA_N, STAGE). Reshape to epi tiles.
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    @cute.jit
    def epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx, sC):
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    @cute.jit
    def epilog_gmem_copy_and_partition(self, tma_atom_c, gC_mnl, epi_tile, sC):
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        sC_for_tma = cute.group_modes(sC, 0, 2)
        gC_for_tma = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1), sC_for_tma, gC_for_tma,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    # -------------------------------------------------------------------
    # Host-side entry point
    # -------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream,  # CUstream
    ):
        """Compile-time entry. Sets up TMA, grid, SMEM, and launches kernel."""
        # Infer dtypes and layouts from tensors
        self.a_dtype = tensor_a.element_type
        self.b_dtype = tensor_b.element_type
        self.c_dtype = tensor_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(tensor_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c)

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.acc_dtype, self.cta_group, self.mma_tiler[:2],
        )

        # TMA load atoms for A and B
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, tensor_a, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, tensor_b, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = a_copy_size + b_copy_size

        # TMA store atom for C
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c, epi_smem_layout, self.epi_tile,
        )

        # Compute K tiles at host level (compile-time constant).
        K = cute.size(tensor_a.layout, mode=[1])
        num_k_tiles = cute.ceil_div(K, self.mma_tiler[2])

        # Grid via persistent tile scheduler.
        M = cute.size(tensor_a.layout, mode=[0])
        N = cute.size(tensor_b.layout, mode=[0])
        num_tiles_m = cute.ceil_div(M, self.cta_tile_shape_mnk[0])
        num_tiles_n = cute.ceil_div(N, self.cta_tile_shape_mnk[1])
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (num_tiles_m, num_tiles_n, 1),
            (*self.cluster_shape_mn, 1),
        )
        grid = tile_sched_params.get_grid_shape(max_active_clusters)

        # SharedStorage
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch kernel
        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            num_k_tiles,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    # -------------------------------------------------------------------
    # GPU kernel
    # -------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        num_k_tiles: cutlass.Constexpr[int],
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # ------ Allocate SMEM + barriers ------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_full_mbar_ptr = storage.ab_full_mbar_ptr.data_ptr()
        ab_empty_mbar_ptr = storage.ab_empty_mbar_ptr.data_ptr()
        acc_full_mbar_ptr = storage.acc_full_mbar_ptr.data_ptr()
        acc_empty_mbar_ptr = storage.acc_empty_mbar_ptr.data_ptr()

        # Init AB barriers (epilogue warp 0)
        if warp_idx == self.epilog_warp_id[0]:
            for s in range(self.num_ab_stage):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(ab_full_mbar_ptr + s, 1)
                    cute.arch.mbarrier_init(ab_empty_mbar_ptr + s, 1)

        # Init ACC barriers (MMA warp)
        if warp_idx == self.mma_warp_id:
            for s in range(self.num_acc_stage):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(acc_full_mbar_ptr + s, 1)
                    cute.arch.mbarrier_init(acc_empty_mbar_ptr + s, 4)

        # Tmem allocator
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster sync after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ------ SMEM tensors ------
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # No multicast for cluster=(1,1)
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        acc_full_mcast_mask = None

        # ------ Partition global and shared tensors ------
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # TMA partitions
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # MMA fragments
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # num_k_tiles is passed as a Constexpr parameter from host

        # Wait for cluster init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # =================================================================
        # TMA WARP
        # =================================================================
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, cute.arch.grid_dim()
            )
            total_k_tile_cnt = cutlass.Int32(0)
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0], cur_tile_coord[1], 0
                )
                cur_k_tile_cnt = num_k_tiles

                # Slice to this tile's M, N position
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                num_prev_k_blk = total_k_tile_cnt
                total_k_tile_cnt += cur_k_tile_cnt

                # TMA load loop
                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    smem_wr_buffer = (num_prev_k_blk + k_tile) % self.num_ab_stage
                    wr_phase = (num_prev_k_blk + k_tile) // self.num_ab_stage % 2 ^ 1

                    # Wait AB empty
                    cute.arch.mbarrier_wait(
                        ab_empty_mbar_ptr + smem_wr_buffer, wr_phase
                    )

                    # Arrive and expect bytes
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full_mbar_ptr + smem_wr_buffer,
                            self.num_tma_load_bytes,
                        )

                    # TMA copy A, B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_tile)],
                        tAsA[(None, smem_wr_buffer)],
                        tma_bar_ptr=ab_full_mbar_ptr + smem_wr_buffer,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_tile)],
                        tBsB[(None, smem_wr_buffer)],
                        tma_bar_ptr=ab_full_mbar_ptr + smem_wr_buffer,
                        mcast_mask=b_full_mcast_mask,
                    )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # =================================================================
        # MMA WARP
        # =================================================================
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, cute.arch.grid_dim()
            )
            total_k_tile_cnt = cutlass.Int32(0)
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_k_tile_cnt = num_k_tiles

                acc_buf_idx = tile_sched.num_tiles_executed % self.num_acc_stage
                tCtAcc = tCtAcc_base[(None, None, None, acc_buf_idx)]

                num_prev_k_blk = total_k_tile_cnt
                total_k_tile_cnt += cur_k_tile_cnt

                # Wait ACC empty
                acc_empty_phase = (
                    tile_sched.num_tiles_executed // self.num_acc_stage % 2 ^ 1
                )
                cute.arch.mbarrier_wait(
                    acc_empty_mbar_ptr + acc_buf_idx, acc_empty_phase
                )

                # Reset accumulate flag
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                # MMA mainloop
                for k_tile in range(cur_k_tile_cnt):
                    smem_rd_buffer = (
                        num_prev_k_blk + k_tile
                    ) % self.num_ab_stage
                    rd_phase = (
                        (num_prev_k_blk + k_tile) // self.num_ab_stage % 2
                    )

                    # Wait AB full
                    cute.arch.mbarrier_wait(
                        ab_full_mbar_ptr + smem_rd_buffer, rd_phase
                    )

                    # Compute
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblock_coord = (None, None, kblock_idx, smem_rd_buffer)
                        cute.gemm(
                            tiled_mma, tCtAcc,
                            tCrA[kblock_coord], tCrB[kblock_coord],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    # Signal AB empty
                    with cute.arch.elect_one():
                        tcgen05.commit(
                            ab_empty_mbar_ptr + smem_rd_buffer,
                            ab_empty_mcast_mask,
                            self.cta_group,
                        )

                # Signal ACC full
                with cute.arch.elect_one():
                    tcgen05.commit(
                        acc_full_mbar_ptr + acc_buf_idx,
                        acc_full_mcast_mask,
                        self.cta_group,
                    )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # =================================================================
        # EPILOGUE WARPS (0-3)
        # =================================================================
        if warp_idx < self.mma_warp_id:
            # Alloc tensor memory
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx

            # Partition for epilogue
            (
                tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, False
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c, bSG_sC, bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile, sC)

            # Persistent tile loop
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (cur_tile_coord[0], cur_tile_coord[1], 0)

                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

                acc_buf_idx = tile_sched.num_tiles_executed % self.num_acc_stage
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_buf_idx)
                ]

                # Wait ACC full
                acc_full_phase = (
                    tile_sched.num_tiles_executed // self.num_acc_stage % 2
                )
                cute.arch.mbarrier_wait(
                    acc_full_mbar_ptr + acc_buf_idx, acc_full_phase
                )

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # Store subtiles
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    # TMEM → RMEM
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Type convert (fp32 accum → bf16)
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    # RMEM → SMEM
                    epi_buffer = (
                        num_prev_subtiles + subtile_idx
                    ) % self.num_epi_stage
                    cute.copy(
                        tiled_copy_r2s, tRS_rC,
                        tRS_sC[(None, None, None, epi_buffer)],
                    )

                    # Fence + sync
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    # SMEM → GMEM via TMA
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, epi_buffer)],
                            bSG_gC[(None, subtile_idx)],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            self.num_epi_stage - 1, read=True
                        )
                    self.epilog_sync_barrier.arrive_and_wait()

                # Signal ACC empty
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(acc_empty_mbar_ptr + acc_buf_idx)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Clean up tmem
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

            # Wait last AB empty
            if warp_idx == self.epilog_warp_id[0]:
                pass  # dense GEMM: K tiles fixed, not tracked here


# ===========================================================================
# Host-side wrapper
# ===========================================================================

_COMPILE_CACHE: dict = {}


@functools.lru_cache(maxsize=32)
def _get_hardware_info():
    hw = utils.HardwareInfo()
    return hw.get_max_active_clusters(1)


def dense_gemm(
    a: torch.Tensor,   # [M, K] bf16, K-contiguous
    b: torch.Tensor,   # [N, K] bf16, K-contiguous
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a from-scratch CuTe DSL dense GEMM on Blackwell.

    C[M,N] = A[M,K] @ B[N,K]^T   (equivalently: A @ B.t())
    """
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    N, Kb = b.shape
    assert K == Kb
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a.stride(1) == 1 and b.stride(1) == 1, "K must be contiguous"

    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)

    # Wrap as 3D (M, K, L=1) cute tensors using from_dlpack to share memory.
    from cutlass.cute.runtime import from_dlpack

    a_cute = from_dlpack(a.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    b_cute = from_dlpack(b.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    c_cute = from_dlpack(out.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )

    max_active_clusters = _get_hardware_info()
    stream = cutlass_torch.default_stream()

    # Cache compiled kernel by (M, N, K) shape + num_k_tiles.
    cache_key = (M, N, K)
    if cache_key not in _COMPILE_CACHE:
        kernel = SimpleDenseGemm()
        compiled = cute.compile(
            kernel, a_cute, b_cute, c_cute, max_active_clusters, stream,
        )
        _COMPILE_CACHE[cache_key] = compiled
    else:
        compiled = _COMPILE_CACHE[cache_key]

    compiled(a_cute, b_cute, c_cute, max_active_clusters, stream)
    torch.cuda.synchronize()
    return out
