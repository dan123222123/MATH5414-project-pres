### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 32064be8-bd92-4883-a626-eb406b3f4481
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 13702445-99bd-4da6-b211-d4fba8926653
begin
	using LinearAlgebra, GridArrays, CUDA, BenchmarkTools
end

# ╔═╡ df286278-c752-419e-a57c-95d9235dc79b
begin
	n = 16;
	g = 100;
end

# ╔═╡ a24add2d-1de9-4850-aee8-7c1020511f5f
begin
    T = ComplexF32
    A = rand(T, n, n)
    E = rand(T, n, n)
    z = randn(T)
end;

# ╔═╡ 72dfc2f1-4956-415d-a1d2-5064b8efcced
@benchmark z * E - A

# ╔═╡ 38e2aee9-89aa-4fb2-91a2-a7dd801f3ced
begin
    gA = cu(A)
    gE = cu(E)
end

# ╔═╡ c1ba2101-8896-45c6-90f2-fc25871202e4
begin
    function ccclean()
        CUDA.memory_status()
        GC.gc()
        CUDA.reclaim()
        CUDA.memory_status()
        synchronize()
    end
    ccclean()
end

# ╔═╡ 12bfa915-a429-42d2-89c9-e17df430814b
@benchmark @sync z * gE - gA

# ╔═╡ 68d734af-d5e5-42c6-a582-3c00ca076e14
ccclean()

# ╔═╡ 254d855a-93e8-4cae-b572-b4bd2afa5a79
begin
    gx = EquispacedGrid(g, -1, 1)
    grid = ProductGrid(gx, gx * 1im)
    zg = Matrix{T}(sum.(collect(grid))) # matrix
    zv = collect(Iterators.flatten(zg)) #vector
end

# ╔═╡ 8e759d67-c8e0-49d1-9165-eae1a4c187ed
zv .* Ref(E) .- Ref(A)

# ╔═╡ cba29757-de70-4a6f-a487-0af33ca50a7b
@benchmark zv .* Ref(E) .- Ref(A)

# ╔═╡ bd6ab2a1-c1b9-4759-840c-e9e5c61a0790
begin
    ccclean()
    @benchmark @sync zv .* Ref(gE) .- Ref(gA)
end

# ╔═╡ fe32dd50-3809-4a58-9e4b-a419f405806e
CUDA.@profile zv .* Ref(gE) .- Ref(gA)

# ╔═╡ Cell order:
# ╠═32064be8-bd92-4883-a626-eb406b3f4481
# ╠═13702445-99bd-4da6-b211-d4fba8926653
# ╠═df286278-c752-419e-a57c-95d9235dc79b
# ╠═a24add2d-1de9-4850-aee8-7c1020511f5f
# ╠═72dfc2f1-4956-415d-a1d2-5064b8efcced
# ╠═38e2aee9-89aa-4fb2-91a2-a7dd801f3ced
# ╠═c1ba2101-8896-45c6-90f2-fc25871202e4
# ╠═12bfa915-a429-42d2-89c9-e17df430814b
# ╟─68d734af-d5e5-42c6-a582-3c00ca076e14
# ╠═254d855a-93e8-4cae-b572-b4bd2afa5a79
# ╠═8e759d67-c8e0-49d1-9165-eae1a4c187ed
# ╠═cba29757-de70-4a6f-a487-0af33ca50a7b
# ╠═bd6ab2a1-c1b9-4759-840c-e9e5c61a0790
# ╠═fe32dd50-3809-4a58-9e4b-a419f405806e
