import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sympify, lambdify
import math  # Added for factorial function

# ---------------------------------------------
# ODE Methods
# ---------------------------------------------
def get_ode_function(expr):
    """Convert string expression to a function f(x, y)"""
    x, y = sp.symbols('x y')
    f_expr = sympify(expr)
    f_func = lambdify((x, y), f_expr, modules='numpy')
    return f_func

def taylor_series_method(f, x0, y0, h, n):
    results = []
    for _ in range(n):
        f0 = f(x0, y0)
        y1 = y0 + h * f0
        results.append((x0 + h, y1))
        x0 += h
        y0 = y1
    return results

def euler_method(f, x0, y0, h, n):
    results = []
    for _ in range(n):
        y0 = y0 + h * f(x0, y0)
        x0 = x0 + h
        results.append((x0, y0))
    return results

def runge_kutta_method(f, x0, y0, h, n):
    results = []
    for _ in range(n):
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + 0.5 * h, y0 + 0.5 * k1)
        k3 = h * f(x0 + 0.5 * h, y0 + 0.5 * k2)
        k4 = h * f(x0 + h, y0 + k3)
        y0 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x0 = x0 + h
        results.append((x0, y0))
    return results

def milne_method(f, x0, y0, h, n):
    init_steps = runge_kutta_method(f, x0, y0, h, 3)
    results = [(x0, y0)] + init_steps
    
    output = []
    for i in range(3, n):
        x3, y3 = results[i]
        x2, y2 = results[i-1]
        x1, y1 = results[i-2]
        x0, y0 = results[i-3]
        
        y_pred = y0 + (4*h/3)*(2*f(x3,y3) - f(x2,y2) + 2*f(x1,y1))
        x_new = x3 + h
        
        y_corr = y2 + (h/3)*(f(x2,y2) + 4*f(x3,y3) + f(x_new,y_pred))
        
        results.append((x_new, y_corr))
        output.append((x_new, y_pred, y_corr))
    return output

def adams_bashforth_method(f, x0, y0, h, n):
    init_steps = runge_kutta_method(f, x0, y0, h, 3)
    results = [(x0, y0)] + init_steps
    
    output = []
    for i in range(3, n):
        x3, y3 = results[i]
        x2, y2 = results[i-1]
        x1, y1 = results[i-2]
        x0, y0 = results[i-3]
        
        y_pred = y3 + (h/24)*(55*f(x3,y3) - 59*f(x2,y2) + 37*f(x1,y1) - 9*f(x0,y0))
        x_new = x3 + h
        
        y_corr = y3 + (h/24)*(9*f(x_new,y_pred) + 19*f(x3,y3) - 5*f(x2,y2) + f(x1,y1))
        
        results.append((x_new, y_corr))
        output.append((x_new, y_pred, y_corr))
    return output

# ---------------------------------------------
# Interpolation Methods
# ---------------------------------------------
def lagrange_interpolation(x_data, y_data, x_eval):
    n = len(x_data)
    result = 0.0
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x_eval - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

def newton_divided_difference(x_data, y_data, x_eval):
    n = len(x_data)
    coef = np.zeros([n, n])
    coef[:, 0] = y_data
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_data[i+j] - x_data[i])
    
    result = coef[0][0]
    for j in range(1, n):
        term = coef[0][j]
        for k in range(j):
            term *= (x_eval - x_data[k])
        result += term
    return result

def newton_forward_difference(x_data, y_data, x_eval):
    n = len(x_data)
    h = x_data[1] - x_data[0]
    u = (x_eval - x_data[0]) / h
    
    forward_diff = np.zeros([n, n])
    forward_diff[:, 0] = y_data
    
    for j in range(1, n):
        for i in range(n - j):
            forward_diff[i][j] = forward_diff[i+1][j-1] - forward_diff[i][j-1]
    
    result = forward_diff[0][0]
    for j in range(1, n):
        term = forward_diff[0][j]
        for k in range(j):
            term *= (u - k)
        term /= math.factorial(j)  # Fixed: Using math.factorial instead of np.math.factorial
        result += term
    return result

def newton_backward_difference(x_data, y_data, x_eval):
    n = len(x_data)
    h = x_data[1] - x_data[0]
    u = (x_eval - x_data[-1]) / h
    
    backward_diff = np.zeros([n, n])
    backward_diff[:, 0] = y_data
    
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            backward_diff[i][j] = backward_diff[i][j-1] - backward_diff[i-1][j-1]
    
    result = backward_diff[-1][0]
    for j in range(1, n):
        term = backward_diff[-1][j]
        for k in range(j):
            term *= (u + k)
        term /= math.factorial(j)  # Fixed: Using math.factorial instead of np.math.factorial
        result += term
    return result

# ---------------------------------------------
# Numerical Integration Methods
# ---------------------------------------------
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

def simpsons_13_rule(f, a, b, n):
    if n % 2 != 0:
        n += 1  # Simpson's rule requires even number of intervals
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

def double_integration_trapezoidal(f, x0, xn, y0, yn, nx, ny):
    hx = (xn - x0) / nx
    hy = (yn - y0) / ny
    
    integral = 0
    for i in range(nx + 1):
        x = x0 + i * hx
        for j in range(ny + 1):
            y = y0 + j * hy
            weight = 1
            if i == 0 or i == nx:
                weight *= 0.5
            if j == 0 or j == ny:
                weight *= 0.5
            integral += weight * f(x, y)
    integral *= hx * hy
    return integral

def double_integration_simpsons(f, x0, xn, y0, yn, nx, ny):
    if nx % 2 != 0:
        nx += 1
    if ny % 2 != 0:
        ny += 1
    
    hx = (xn - x0) / nx
    hy = (yn - y0) / ny
    
    integral = 0
    for i in range(nx + 1):
        x = x0 + i * hx
        wx = 1
        if i > 0 and i < nx:
            wx = 4 if i % 2 == 1 else 2
        for j in range(ny + 1):
            y = y0 + j * hy
            wy = 1
            if j > 0 and j < ny:
                wy = 4 if j % 2 == 1 else 2
            integral += wx * wy * f(x, y)
    integral *= (hx * hy) / 9
    return integral

# ---------------------------------------------
# Matrix Input Helper
# ---------------------------------------------
def display_matrix_input(n, m, key_prefix):
    cols = st.columns(m)
    matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            entry = cols[j].number_input(
                f"{key_prefix} Row {i+1}, Col {j+1}", key=f"{key_prefix}{i}{j}")
            row.append(entry)
        matrix.append(row)
    return np.array(matrix)

# ---------------------------------------------
# Main App
# ---------------------------------------------
st.set_page_config(page_title="Numerical Methods Calculator", layout="wide")

st.title("🔢 Unified Numerical Methods Calculator")

section = st.sidebar.radio(
    "📘 Choose Section", 
    ["Numerical Solution of ODEs", 
     "Equations & Eigenvalue Problems",
     "Interpolation Methods",
     "Numerical Integration"]
)

# =============================================
# Numerical ODEs Section
# =============================================
if section == "Numerical Solution of ODEs":
    st.subheader("🧮 Solve ODEs Numerically")

    method = st.sidebar.selectbox(
        "Choose a Numerical Method",
        ["Taylor Series Method", "Euler's Method", "Runge-Kutta Method (4th Order)",
         "Milne's Method (Predictor-Corrector)", "Adams-Bashforth Method (Predictor-Corrector)"])

    ode_expr = st.text_input("Enter ODE dy/dx = f(x, y)", "x + y", 
                            help="Use Python syntax with x and y variables, e.g., x**2 + y, sin(x)*cos(y), etc.")
    
    col1, col2 = st.columns(2)
    x0 = col1.number_input("Initial x (x₀)", value=0.0)
    y0 = col2.number_input("Initial y (y₀)", value=1.0)
    
    col1, col2 = st.columns(2)
    h = col1.number_input("Step Size (h)", value=0.1, format="%.4f")
    n = col2.number_input("Number of Steps (n)", value=10, step=1)

    if st.button("🚀 Solve ODE"):
        try:
            f = get_ode_function(ode_expr)
            
            with st.spinner("Calculating..."):
                if method == "Taylor Series Method":
                    results = taylor_series_method(f, x0, y0, h, int(n))
                    df = pd.DataFrame(results, columns=["x", "y"])
                elif method == "Euler's Method":
                    results = euler_method(f, x0, y0, h, int(n))
                    df = pd.DataFrame(results, columns=["x", "y"])
                elif method == "Runge-Kutta Method (4th Order)":
                    results = runge_kutta_method(f, x0, y0, h, int(n))
                    df = pd.DataFrame(results, columns=["x", "y"])
                elif method == "Milne's Method (Predictor-Corrector)":
                    results = milne_method(f, x0, y0, h, int(n))
                    df = pd.DataFrame(results, columns=["x", "Predictor y", "Corrector y"])
                elif method == "Adams-Bashforth Method (Predictor-Corrector)":
                    results = adams_bashforth_method(f, x0, y0, h, int(n))
                    df = pd.DataFrame(results, columns=["x", "Predictor y", "Corrector y"])

                st.success("✅ Computation Completed!")
                st.dataframe(df.style.format("{:.6f}"))

                st.subheader("📈 Plot")
                fig, ax = plt.subplots(figsize=(8, 5))
                if "Predictor y" in df.columns:
                    ax.plot(df["x"], df["Predictor y"], 'bo--', label="Predictor")
                    ax.plot(df["x"], df["Corrector y"], 'gs-', label="Corrector")
                else:
                    ax.plot(df["x"], df["y"], 'r-o', label="y(x)")
                ax.legend()
                ax.set_title(f"Solution of dy/dx = {ode_expr} using {method}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.grid(True)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Please check your ODE expression. Use valid Python syntax with x and y variables.")

# =============================================
# Equation Solving and Eigenvalue Section
# =============================================
elif section == "Equations & Eigenvalue Problems":
    st.subheader("🧠 Solve Equations and Matrix Problems")

    method = st.sidebar.selectbox("Choose Method", [
        "Newton-Raphson Method",
        "Gauss Elimination",
        "Gauss-Jordan Elimination",
        "Gauss-Seidel Iterative",
        "Power Method (Eigenvalues)"
    ])

    if method == "Newton-Raphson Method":
        expr = st.text_input("Enter function f(x)", "x**2 - 2")
        x0 = st.number_input("Initial guess x₀", value=1.0)
        tol = st.number_input("Tolerance", value=1e-6, format="%e")
        max_iter = st.number_input("Max iterations", value=100, step=10)
        if st.button("🔍 Find Root"):
            try:
                x = sp.symbols('x')
                f = sympify(expr)
                f_prime = f.diff(x)
                f_lambda = lambdify(x, f, modules='numpy')
                f_prime_lambda = lambdify(x, f_prime, modules='numpy')

                for i in range(int(max_iter)):
                    fx = f_lambda(x0)
                    fpx = f_prime_lambda(x0)
                    if fpx == 0:
                        st.error("Zero derivative. No solution found.")
                        break
                    x1 = x0 - fx / fpx
                    if abs(x1 - x0) < tol:
                        st.success(f"Root found: {x1:.6f} in {i+1} iterations")
                        break
                    x0 = x1
                else:
                    st.warning("Max iterations reached without convergence.")
            except Exception as e:
                st.error(f"Error: {e}")

    elif method in ["Gauss Elimination", "Gauss-Jordan Elimination", "Gauss-Seidel Iterative", "Power Method (Eigenvalues)"]:
        n = st.number_input("Matrix Size (n x n)", min_value=2, max_value=10, value=3)
        st.markdown("*Enter Matrix A:*")
        A = display_matrix_input(int(n), int(n), "A")
        if method != "Power Method (Eigenvalues)":
            st.markdown("*Enter Vector b:*")
            b = display_matrix_input(int(n), 1, "b")

        if st.button("🧮 Solve Matrix"):
            try:
                if method == "Gauss Elimination":
                    x = np.linalg.solve(A, b)
                    st.success("Solution Vector x:")
                    st.write(x)
                elif method == "Gauss-Jordan Elimination":
                    Ab = np.hstack([A, b])
                    Ab = Ab.astype(float)
                    n = len(Ab)
                    for i in range(n):
                        Ab[i] = Ab[i] / Ab[i, i]
                        for j in range(n):
                            if i != j:
                                Ab[j] = Ab[j] - Ab[j, i] * Ab[i]
                    x = Ab[:, -1]
                    st.success("Solution Vector x:")
                    st.write(x)
                elif method == "Gauss-Seidel Iterative":
                    x = np.zeros_like(b)
                    for _ in range(100):
                        x_new = np.copy(x)
                        for i in range(int(n)):
                            s1 = sum(A[i][j] * x_new[j] for j in range(i))
                            s2 = sum(A[i][j] * x[j] for j in range(i + 1, int(n)))
                            x_new[i] = (b[i] - s1 - s2) / A[i][i]
                        if np.allclose(x, x_new, atol=1e-6):
                            break
                        x = x_new
                    st.success("Solution Vector x:")
                    st.write(x)
                elif method == "Power Method (Eigenvalues)":
                    x = np.ones((int(n), 1))
                    for _ in range(100):
                        x_next = A @ x
                        x_next = x_next / np.linalg.norm(x_next)
                        if np.allclose(x, x_next, atol=1e-6):
                            break
                        x = x_next
                    eigenvalue = float((x.T @ A @ x) / (x.T @ x))
                    st.success(f"Dominant Eigenvalue: {eigenvalue:.6f}")
                    st.write("Corresponding Eigenvector:")
                    st.write(x)
            except Exception as e:
                st.error(f"Error: {e}")

# =============================================
# Interpolation Section
# =============================================
elif section == "Interpolation Methods":
    st.subheader("📊 Interpolation Methods")
    
    method = st.sidebar.selectbox(
        "Choose Interpolation Method",
        ["Lagrange's Interpolation", 
         "Newton's Divided Difference", 
         "Newton's Forward Difference", 
         "Newton's Backward Difference"]
    )
    
    st.markdown("### Enter Data Points")
    n = st.number_input("Number of data points", min_value=2, max_value=10, value=4)
    
    x_data = []
    y_data = []
    cols = st.columns(2)
    for i in range(n):
        x = cols[0].number_input(f"x_{i}", value=float(i), key=f"x_{i}")
        y = cols[1].number_input(f"y_{i}", value=float(i**2), key=f"y_{i}")
        x_data.append(x)
        y_data.append(y)
    
    x_eval = st.number_input("Point to evaluate (x)", value=(x_data[0] + x_data[-1])/2)
    
    if st.button("🚀 Interpolate"):
        with st.spinner("Calculating..."):
            if method == "Lagrange's Interpolation":
                result = lagrange_interpolation(x_data, y_data, x_eval)
            elif method == "Newton's Divided Difference":
                result = newton_divided_difference(x_data, y_data, x_eval)
            elif method == "Newton's Forward Difference":
                result = newton_forward_difference(x_data, y_data, x_eval)
            elif method == "Newton's Backward Difference":
                result = newton_backward_difference(x_data, y_data, x_eval)
            
            st.success(f"Interpolated value at x = {x_eval}: *{result:.6f}*")
            
            # Plotting
            x_plot = np.linspace(min(x_data), max(x_data), 100)
            y_plot = []
            for x in x_plot:
                if method == "Lagrange's Interpolation":
                    y_plot.append(lagrange_interpolation(x_data, y_data, x))
                elif method == "Newton's Divided Difference":
                    y_plot.append(newton_divided_difference(x_data, y_data, x))
                elif method == "Newton's Forward Difference":
                    y_plot.append(newton_forward_difference(x_data, y_data, x))
                elif method == "Newton's Backward Difference":
                    y_plot.append(newton_backward_difference(x_data, y_data, x))
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_data, y_data, 'ro', label="Data Points")
            ax.plot(x_plot, y_plot, 'b-', label="Interpolation")
            ax.plot(x_eval, result, 'gs', label=f"Interpolated Point ({x_eval:.2f}, {result:.2f})")
            ax.legend()
            ax.set_title(f"{method}")
            ax.grid(True)
            st.pyplot(fig)

# =============================================
# Numerical Integration Section
# =============================================
elif section == "Numerical Integration":
    st.subheader("📐 Numerical Integration Methods")
    
    method = st.sidebar.selectbox(
        "Choose Integration Method",
        ["Single: Trapezoidal Rule", 
         "Single: Simpson's 1/3 Rule", 
         "Double: Trapezoidal Rule", 
         "Double: Simpson's 1/3 Rule"]
    )
    
    if "Single" in method:
        st.markdown("### Enter Function f(x)")
        func_expr = st.text_input("Function (use Python syntax, e.g., x*2 + np.sin(x))", "x*2")
        
        a = st.number_input("Lower limit (a)", value=0.0)
        b = st.number_input("Upper limit (b)", value=1.0)
        n = st.number_input("Number of intervals (n)", min_value=1, value=100)
        
        if st.button("🚀 Integrate"):
            try:
                f = lambda x: eval(func_expr)
                if "Trapezoidal" in method:
                    result = trapezoidal_rule(f, a, b, int(n))
                elif "Simpson's" in method:
                    result = simpsons_13_rule(f, a, b, int(n))
                
                st.success(f"Integral result: *{result:.6f}*")
                
                # Plotting
                x_plot = np.linspace(a, b, 1000)
                y_plot = f(x_plot)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x_plot, y_plot, 'r-', label=f"f(x) = {func_expr}")
                ax.fill_between(x_plot, y_plot, alpha=0.3)
                ax.set_title(f"Numerical Integration using {method}")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif "Double" in method:
        st.markdown("### Enter Function f(x, y)")
        func_expr = st.text_input("Function (e.g., x*2 + y2)", "x2 + y*2")
        
        x0 = st.number_input("x lower limit (x0)", value=0.0)
        xn = st.number_input("x upper limit (xn)", value=1.0)
        y0 = st.number_input("y lower limit (y0)", value=0.0)
        yn = st.number_input("y upper limit (yn)", value=1.0)
        nx = st.number_input("x intervals (nx)", min_value=1, value=10)
        ny = st.number_input("y intervals (ny)", min_value=1, value=10)
        
        if st.button("🚀 Integrate"):
            try:
                f = lambda x, y: eval(func_expr)
                if "Trapezoidal" in method:
                    result = double_integration_trapezoidal(f, x0, xn, y0, yn, int(nx), int(ny))
                elif "Simpson's" in method:
                    result = double_integration_simpsons(f, x0, xn, y0, yn, int(nx), int(ny))
                
                st.success(f"Double Integral result: *{result:.6f}*")
            except Exception as e:
                st.error(f"Error: {e}")

# =============================================
# Instructions Section
# =============================================
st.sidebar.subheader("ℹ Instructions")
st.sidebar.markdown("""
### 📌 Getting Started
1. Select a section using the sidebar
2. Choose a numerical method
3. Enter the required inputs
4. Click the compute button to view results

### 🧮 Numerical ODE Solvers
- Solve dy/dx = f(x,y) using various methods
- Includes Euler, Runge-Kutta, and predictor-corrector methods

### 🔢 Equation Solvers
- Find roots of equations
- Solve linear systems
- Compute eigenvalues

### 📊 Interpolation Methods
- Estimate values between known data points
- Includes Lagrange and Newton interpolation

### 📐 Numerical Integration
- Approximate definite integrals
- Single and double integration methods
""")

# =============================================
# Footer
# =============================================
st.caption(
    "Numerical Methods Calculator | Team NumeriSolve | "
    "An educational tool for numerical computations"
)