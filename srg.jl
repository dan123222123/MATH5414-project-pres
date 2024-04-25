### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 905d572c-030e-11ef-1cba-bfda5da9ff58
using DrWatson

# ╔═╡ ee6acf8e-3d54-4447-a6d9-f913c0287cf3
@quickactivate "MATH5414-project"

# ╔═╡ df906991-eb40-49bc-9cdd-6172a12dc747
begin
	using LinearAlgebra, GridArrays, CUDA, Adapt, NVTX, ThreadsX, Krylov, BenchmarkTools, MatrixDepot;
	
	CUDA.allowscalar(false);
	
	function ccclean()
		CUDA.memory_status()
		GC.gc()
		CUDA.reclaim()
		CUDA.memory_status()
		synchronize()
	end
end

# ╔═╡ 2b16db44-fb93-4d58-913b-b40e3425133a
using Plots

# ╔═╡ 281c0899-d94a-42bd-b0e7-7253ca380ed1
begin
	n = 5; g = 300
	
	T = ComplexF64
	A = MatrixDepot.grcar(T,n)
	#A = randn(T,n,n)
	E = I

	gx = EquispacedGrid(g, -4, 4)
	grid = ProductGrid(gx, gx * 1im)
	gr = collect(real.(grid.grids[1]))
	gi = collect(imag.(grid.grids[2]))
	
	zg = Matrix{T}(sum.(collect(grid))) # matrix
	zv = collect(Iterators.flatten(zg)) #vector
end;

# ╔═╡ 520608aa-5e74-4f3c-88d6-2ea890da2093
function csrsvdB(zg, A, E, γ, δ)
  mpB = stack(zg .* Ref(E) .- Ref(A), dims=3)
  T1 = (γ .+ (δ .* abs.(zg)))
  T2 = reshape(ThreadsX.collect(svdvals(mpB[:, :, i])[end] for i in axes(mpB, 3)), size(zg))
  return (T1 .* T2)
end

# ╔═╡ bd57df8a-c71f-4ef7-8dc0-76dda1bbee50
srg = csrsvdB(zg, A, E, 1, 1)

# ╔═╡ 12c9a54f-33fa-4bc3-b943-9d56e3ba945b
contourf(gr, gi, log10.(srg)'; color= cgrad(:jet))

# ╔═╡ 841ee133-d4f9-4004-9174-ba72b2a7c11e
@benchmark csrsvdB(zg, A, E, 1, 1)

# ╔═╡ 4d2e4b65-16f5-42e5-8642-0a4292157b64
function gsrsvdB(zg, A, E, γ, δ; alg=CUDA.CUSOLVER.JacobiAlgorithm())
  mpB = cu(stack(zg .* Ref(E) .- Ref(A), dims=3))
  F = CUDA.svd(mpB; alg)
  T1 = (γ .+ (δ .* (abs.(zg))))
  T2 = reshape(Matrix(F.S)[end, :], size(zg))
  return (T1 .* T2)
end

# ╔═╡ e98b81a1-3216-49b6-9c93-12543e91ec5a
begin
	ccclean()
	gsrg = gsrsvdB(zg, A, E, 1, 1);
end

# ╔═╡ df7cdaa2-b55f-4c56-b5fb-349ec59b8f9a
contourf(gr, gi, log10.(gsrg)'; color= cgrad(:jet))

# ╔═╡ c4855c17-1fc2-42d2-be7b-93768fbd4619
begin
	ccclean()
	CUDA.@profile gsrsvdB(zg, A, E, 1, 1; alg=CUDA.CUSOLVER.JacobiAlgorithm())
end

# ╔═╡ f5d7c4d2-be3b-4388-8828-7caad37d267b
begin
	local n = 5;
	local g = 50
	
	local T = ComplexF64
	local A = MatrixDepot.grcar(T,n)
	local E = I

	local gx = EquispacedGrid(g, -4, 4)
	local grid = ProductGrid(gx, gx * 1im)
	local gr = collect(real.(grid.grids[1]))
	local gi = collect(imag.(grid.grids[2]))
	
	local zg = Matrix{T}(sum.(collect(grid))) # matrix
	local zv = collect(Iterators.flatten(zg)) #vector
	ccclean()
	CUDA.@profile gsrsvdB(zg, A, E, 1, 1; alg=CUDA.CUSOLVER.ApproximateAlgorithm())
end

# ╔═╡ 8d46ebb8-a40f-4ad7-84ad-220e957a6aef
function srgil(zg::Matrix{T}, A, E, γ, δ;
    m=size(A, 1),
    mit=(6 * ceil(Int, log(m))),
    usegpu=false,
    ngpu=1,
    zpg=floor(Int, length(zg) / ngpu)
) where {T<:Complex}

    A = Matrix{T}(A)
    if E == I
        E = Matrix{T}(I, size(A))
    end
    E = Matrix{T}(E)

    A, E, _ = schur(A, E)
    A = factorize(A)
    E = factorize(E)

    zv = collect(Iterators.flatten(zg))
    cHR = Mem.pin([zeros(T, mit, mit) for _ in eachindex(zv)])
    ST = [zeros(T, size(A)) for _ in eachindex(zv)]

    if usegpu
        A = cu(A)
        E = cu(E)
        HR = cu.(cHR)
        ST = cu.(ST)
        R = CUDA.rand(T, m)
    else
        A = adapt(typeof(A), A)
        E = adapt(typeof(E), E)
        HR = cHR
        R = rand(T, m)
    end

    # these batch sizes should be computed
    zvpg_batches = Vector(collect(Iterators.partition(zv, zpg)))
    HRpg_batches = Vector(collect(Iterators.partition(HR, zpg)))
    STpg_batches = Vector(collect(Iterators.partition(ST, zpg)))
    batches = zip(zvpg_batches, STpg_batches, HRpg_batches)

    function computeH!(A, E, zv, ST, HR)
        NVTX.@mark "start gridmats"
		ST .= (zv .* Ref(E)) .- Ref(A)
        ST .= ST ./ adjoint.(inv.(ST))
		synchronize()
        NVTX.@mark "gridmats done"
        NVTX.@mark "start invlancz"
        HR .= map(x -> x[1:mit, 1:mit], map(x -> x[3], hermitian_lanczos.(ST, Ref(R), Ref(mit))))
		synchronize()
        NVTX.@mark "end invlancz"
    end

    NVTX.@mark "start batched H"
    for (i, (zvb, STb, HRb)) in enumerate(batches)
        computeH!(A, E, zvb, STb, HRb)
    end
    NVTX.@mark "end batched H"

    if usegpu
        cHR = Array.(HR)
    end


    sr = zeros(real(T), length(zv))

    sr .= (γ .+ δ .* abs.(zv)) .* sqrt.(abs.(hcat(eigvals.(cHR)...)'[:, 1]))

    return reshape(sr, size(zg))

end
##

# ╔═╡ 47706c85-a4f0-4897-ba5b-0e3f1f562d17
D1 = srgil(zg, A, E, 1, 0);

# ╔═╡ d2f419eb-74f2-4e92-85ff-5ff8665c29c3
contourf(gr, gi, log10.(D1)'; color= cgrad(:jet))

# ╔═╡ b6c8dd18-9ae8-4102-bf23-35dcd25e24b1
begin
	local n = 5;
	local g = 50;
	
	local T = ComplexF32
	local A = MatrixDepot.grcar(T,n)
	local E = I

	local gx = EquispacedGrid(g, -4, 4)
	local grid = ProductGrid(gx, gx * 1im)
	local gr = collect(real.(grid.grids[1]))
	local gi = collect(imag.(grid.grids[2]))
	
	local zg = Matrix{T}(sum.(collect(grid))) # matrix
	local zv = collect(Iterators.flatten(zg)) #vector
	ccclean()
	D2 = srgil(zg, A, E, 1, 0; usegpu=true, mit = 10);
	contourf(gr, gi, log10.(D2)'; color= cgrad(:jet))
end

# ╔═╡ a402f77a-5c55-47d3-81b8-252389f9876e
begin
	local n = 1000;
	local g = 10;
	
	local T = ComplexF32
	local A = randn(T,n,n)
	local E = I

	local gx = EquispacedGrid(g, -1, 1)
	local grid = ProductGrid(gx, gx * 1im)
	gr3 = collect(real.(grid.grids[1]))
	gi3 = collect(imag.(grid.grids[2]))
	
	local zg = Matrix{T}(sum.(collect(grid))) # matrix
	local zv = collect(Iterators.flatten(zg)) #vector
	ccclean()
	D3 = srgil(zg, A, E, 1, 0; usegpu=true, mit = 500)
	ccclean()
	CUDA.@profile srgil(zg, A, E, 1, 0; usegpu=true, mit = 500)
end

# ╔═╡ fb804b5c-a921-4d9e-b4d5-885dc5272a53
contourf(gr3, gi3, log10.(D3)'; color= cgrad(:jet))

# ╔═╡ bda8905e-309d-426c-b4b4-6b59db0cbacd
begin
	local n = 5;
	local g = 50;
	
	local T = ComplexF32
	local A = randn(T,n,n)
	local E = I

	local gx = EquispacedGrid(g, -2, 2)
	local grid = ProductGrid(gx, gx * 1im)
		  gr4 = collect(real.(grid.grids[1]))
		  gi4 = collect(imag.(grid.grids[2]))
	
	local zg = Matrix{T}(sum.(collect(grid))) # matrix
	local zv = collect(Iterators.flatten(zg)) #vector
	ccclean()
	D4 = srgil(zg, A, E, 1, 0; usegpu=true, mit = 10)
	ccclean()
	CUDA.@profile srgil(zg, A, E, 1, 0; usegpu=true, mit = 10)
end

# ╔═╡ 648cfd1d-00a1-414a-90bf-4641bba520d4
contourf(gr4, gi4, log10.(D4)'; color= cgrad(:jet))

# ╔═╡ Cell order:
# ╠═905d572c-030e-11ef-1cba-bfda5da9ff58
# ╠═ee6acf8e-3d54-4447-a6d9-f913c0287cf3
# ╠═df906991-eb40-49bc-9cdd-6172a12dc747
# ╠═281c0899-d94a-42bd-b0e7-7253ca380ed1
# ╠═520608aa-5e74-4f3c-88d6-2ea890da2093
# ╠═bd57df8a-c71f-4ef7-8dc0-76dda1bbee50
# ╠═2b16db44-fb93-4d58-913b-b40e3425133a
# ╠═12c9a54f-33fa-4bc3-b943-9d56e3ba945b
# ╠═841ee133-d4f9-4004-9174-ba72b2a7c11e
# ╠═4d2e4b65-16f5-42e5-8642-0a4292157b64
# ╠═e98b81a1-3216-49b6-9c93-12543e91ec5a
# ╠═df7cdaa2-b55f-4c56-b5fb-349ec59b8f9a
# ╠═c4855c17-1fc2-42d2-be7b-93768fbd4619
# ╠═f5d7c4d2-be3b-4388-8828-7caad37d267b
# ╠═8d46ebb8-a40f-4ad7-84ad-220e957a6aef
# ╠═47706c85-a4f0-4897-ba5b-0e3f1f562d17
# ╠═d2f419eb-74f2-4e92-85ff-5ff8665c29c3
# ╠═b6c8dd18-9ae8-4102-bf23-35dcd25e24b1
# ╠═a402f77a-5c55-47d3-81b8-252389f9876e
# ╠═fb804b5c-a921-4d9e-b4d5-885dc5272a53
# ╠═bda8905e-309d-426c-b4b4-6b59db0cbacd
# ╠═648cfd1d-00a1-414a-90bf-4641bba520d4
