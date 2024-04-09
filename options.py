import tkinter as tk
from tkinter import ttk
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

S0_default = 0.025
v_mem = {}
S_mem = {}
z1_mem = {}


def heston_price(S0, K, T, r, kappa, theta, xi, rho, v0, num_paths, num_steps):
    dt = 10 / num_steps
    sqrt_dt = np.sqrt(dt)

    if (rho, kappa, theta, xi) not in v_mem:
        # Generate correlated Brownian motions
        z1 = np.random.normal(size=(num_steps, num_paths))
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(
            size=(num_steps, num_paths)
        )
        v = np.zeros((num_steps + 1, num_paths))
        S = np.zeros((num_steps + 1, num_paths))
        S[0] = S0
        v[0] = v0
        for i in range(1, num_steps + 1):
            # Euler discretization for volatility
            v[i] = np.maximum(
                v[i - 1]
                + kappa * (theta - v[i - 1]) * dt
                + xi * np.sqrt(v[i - 1]) * sqrt_dt * z2[i - 1],
                0,
            )
            S[i] = (
                S[i - 1]
                + r * S[i - 1] * dt
                + np.sqrt(v[i - 1] * dt) * S[i - 1] * z1[i - 1]
            )
        v_mem[(rho, kappa, theta, xi)] = v
        S_mem[(rho, r, kappa, theta, xi)] = S
        z1_mem[(rho, kappa, theta, xi)] = z1

    elif (rho, r, kappa, theta, xi) not in S_mem:
        z1 = z1_mem[(rho, kappa, theta, xi)]
        v = v_mem[(rho, kappa, theta, xi)]
        S = np.zeros((num_steps + 1, num_paths))
        S[0] = S0
        for i in range(1, num_steps + 1):
            S[i] = S[i - 1] + r * S[i - 1] * dt + np.sqrt(v[i - 1] * dt) * S[i-1] * z1[i - 1]
        S_mem[(rho, r, kappa, theta, xi)] = S

    # Calculate option payoff
    payoff = np.maximum(S_mem[(rho, r, kappa, theta, xi)][num_steps] - K, 0)

    # Discounted option price
    option_price = np.exp(-r * T) * np.mean(payoff)

    return option_price


def black_scholes_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def update_plot(*args):
    # Get parameter values from sliders
    params = {
        "T": 10,
        "r": r_slider.get(),
        "kappa": kappa_slider.get(),
        "theta": theta_slider.get(),
        "xi": xi_slider.get(),
        "rho": rho_slider.get(),
        "v0": theta_slider.get()
    }

    # Calculate option prices for different strikes
    K_values = np.linspace(0.5 * S0_default, 1.5 * S0_default, 100)
    option_prices = [
        heston_price(K=K, S0=S0_default, **params, num_paths=100000, num_steps=200)
        for K in K_values
    ]

    black_scholes_prices = black_scholes_price(S0_default, K_values, params["T"], params["r"], np.sqrt(params["v0"]))

    # Clear previous plot
    ax.clear()

    # Plot option price against strike
    ax.plot(K_values, option_prices, label="Heston Model")
    ax.plot(K_values, black_scholes_prices, label="Black Scholes")
    ax.set_title("Option Price vs Strike")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Option Price")
    ax.legend()
    ax.grid(True)



    canvas.draw()


# Create Tkinter window
root = tk.Tk()
root.title("Heston Model Option Price")

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Arrange sliders on the right side
sliders_frame = tk.Frame(root)
sliders_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

# Create sliders for parameters
r_slider = tk.Scale(
    sliders_frame,
    from_=0,
    to=0.1,
    orient=tk.HORIZONTAL,
    label="r",
    length=300,
    resolution=0.001,
    command=update_plot,
)
kappa_slider = tk.Scale(
    sliders_frame,
    from_=0,
    to=20,
    orient=tk.HORIZONTAL,
    label="kappa",
    length=300,
    resolution=1,
    command=update_plot,
)
theta_slider = tk.Scale(
    sliders_frame,
    from_=0,
    to=0.01,
    orient=tk.HORIZONTAL,
    label="theta",
    length=300,
    resolution=0.0001,
    command=update_plot,
)
xi_slider = tk.Scale(
    sliders_frame,
    from_=0,
    to=1,
    orient=tk.HORIZONTAL,
    label="xi",
    length=300,
    resolution=0.01,
    command=update_plot,
)
rho_slider = tk.Scale(
    sliders_frame,
    from_=-0.9,
    to=0.9,
    orient=tk.HORIZONTAL,
    label="rho",
    length=300,
    resolution=0.01,
    command=update_plot,
)

r_slider.set(0.035)
kappa_slider.set(5)
theta_slider.set(0.0055)
xi_slider.set(0.30)
rho_slider.set(-0.29)

# Pack sliders
r_slider.pack(fill=tk.X)
kappa_slider.pack(fill=tk.X)
theta_slider.pack(fill=tk.X)
xi_slider.pack(fill=tk.X)
rho_slider.pack(fill=tk.X)

# Run Tkinter event loop
root.mainloop()
