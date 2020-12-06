using NLsolve

poly(x, p) = sum(p[n] * x^(n-1) for n in 1:length(p))
∂poly(x, p) = sum(n * p[n+1] * x^(n-1) for n in 1:length(p)-1)

"""
        fit_cubic(p1, p2, s1, s2)

Return the coeficients c such that f(x) = c₁ + c₂x + c₃x² + c₄x³ is a cubic polynomial that passes through the points p1 = (x1, y1) and p2 = (x2, y2) with slope s1 at p1 and slope s2 at p2.
"""
function fit_cubic(p1, p2, s1, s2)
    x1, y1 = p1
    x2, y2 = p2

    function f!(F, c)
        F[1] = poly(x1, c) - y1
        F[2] = poly(x2, c) - y2
        F[3] = ∂poly(x1, c) - s1
        F[4] = ∂poly(x2, c) - s2
    end

    results = nlsolve(f!, zeros(4))
    return results.zero
end

