import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scienceplots
from modal_analysis import modal_analysis, simulation, fft


N = 2
s_period = 1e-2


plt.style.use([
    'grid',
    'notebook'
])


class HalfCarModelWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Half-Car Model")
        
        # Create labels and entry widgets for the input parameters
        tk.Label(self, text="Mass M1:").grid(row=0, column=0, padx=5, pady=5)
        self.mass_m1_entry = tk.Entry(self)
        self.mass_m1_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self, text="Mass M2:").grid(row=1, column=0, padx=5, pady=5)
        self.mass_m2_entry = tk.Entry(self)
        self.mass_m2_entry.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(self, text="Spring Constant K1:").grid(row=2, column=0, padx=5, pady=5)
        self.spring_k1_entry = tk.Entry(self)
        self.spring_k1_entry.grid(row=2, column=1, padx=5, pady=5)
        
        tk.Label(self, text="Spring Constant K2:").grid(row=3, column=0, padx=5, pady=5)
        self.spring_k2_entry = tk.Entry(self)
        self.spring_k2_entry.grid(row=3, column=1, padx=5, pady=5)
        
        tk.Label(self, text="Damping Constant C1:").grid(row=4, column=0, padx=5, pady=5)
        self.damping_c1_entry = tk.Entry(self)
        self.damping_c1_entry.grid(row=4, column=1, padx=5, pady=5)
        
        tk.Label(self, text="Damping Constant C2:").grid(row=5, column=0, padx=5, pady=5)
        self.damping_c2_entry = tk.Entry(self)
        self.damping_c2_entry.grid(row=5, column=1, padx=5, pady=5)

        tk.Label(self, text="Forced Frequency:").grid(row=6, column=0, padx=5, pady=5)
        self.forced_freq_entry = tk.Entry(self)
        self.forced_freq_entry.grid(row=6, column=1, padx=5, pady=5)

        tk.Label(self, text="Forced Amplitude []:").grid(row=7, column=0, padx=5, pady=5)
        self.forced_amp_entry = tk.Entry(self)
        self.forced_amp_entry.grid(row=7, column=1, padx=5, pady=5)

        tk.Label(self, text="B [[]]:").grid(row=8, column=0, padx=5, pady=5)
        self.B_entry = tk.Entry(self)
        self.B_entry.grid(row=8, column=1, padx=5, pady=5)

        tk.Label(self, text="X0 []:").grid(row=9, column=0, padx=5, pady=5)
        self.X0_entry = tk.Entry(self)
        self.X0_entry.grid(row=9, column=1, padx=5, pady=5)

        tk.Label(self, text="V0 []:").grid(row=10, column=0, padx=5, pady=5)
        self.V0_entry = tk.Entry(self)
        self.V0_entry.grid(row=10, column=1, padx=5, pady=5)
        
        # Create a button to compute some things
        tk.Button(self, text="Compute", command=self.compute).grid(row=11, column=0, columnspan=2, padx=5, pady=5)
        
    def compute(self):
        plt.close('all')
        
        # close previous window
        for widget in self.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy()

        # Get the input parameter values from the entry widgets
        mass_m1    = float(self.mass_m1_entry.get())
        mass_m2    = float(self.mass_m2_entry.get())
        spring_k1  = float(self.spring_k1_entry.get())
        spring_k2  = float(self.spring_k2_entry.get())
        damping_c1 = float(self.damping_c1_entry.get())
        damping_c2 = float(self.damping_c2_entry.get())

        wf = float(self.forced_freq_entry.get())

        f = np.zeros((N,1))
        B = np.zeros((N,N))
        X0 = np.array([1, 0])
        V0 = np.array([0, 1])

        if wf != 0:
            f  = np.fromstring(self.forced_amp_entry.get(), dtype=float, sep=' ').reshape(N,1)
            B  = np.fromstring(self.B_entry.get(), dtype=float, sep=' ').reshape(N,N)
            X0 = np.fromstring(self.X0_entry.get(), dtype=float, sep=' ').reshape(N,1)
            V0 = np.fromstring(self.V0_entry.get(), dtype=float, sep=' ').reshape(N,1)
        
        M = np.array([[mass_m1, 0], [0, mass_m2]])
        K = np.array([[spring_k1 + spring_k2, -spring_k2], [-spring_k2, spring_k2]])
        C = np.array([[damping_c1 + damping_c2, -damping_c2], [-damping_c2, damping_c2]])

        # #

        x, wn, zetas, wd, P, _, _, _, x_exprs = modal_analysis(
            M,
            K,
            C,
            X0=X0,
            V0=V0,
            f=f,
            B=B,
            wf=wf,
        )

        # simulation
        t, x_t = simulation(
            x,
            60,
            s_period
        )   

        # #

        # Do some computations
        # ...

        
        # Create a new window to display the results
        results_window = tk.Toplevel(self)
        results_window.title("Half-Car Model Results")
        results_window.geometry("1360x225")
        np.array2string
        # Create labels to display the matrices
        tk.Label(results_window, text="Wn:").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(results_window, text=np.array2string(wn, precision=3)).grid(row=0, column=1, padx=5, pady=5)
    
        tk.Label(results_window, text="Wd:").grid(row=1, column=0, padx=5, pady=5)
        tk.Label(results_window, text=np.array2string(wd, precision=3)).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(results_window, text="Zetas:").grid(row=2, column=0, padx=5, pady=5)
        tk.Label(results_window, text=np.array2string(zetas, precision=3)).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(results_window, text="P:").grid(row=3, column=0, padx=5, pady=5)
        tk.Label(results_window, text=P).grid(row=3, column=1, padx=5, pady=5)

        for i in range(N):
            tk.Label(results_window, text=f"x{i+1}(t):").grid(row=6+i, column=0, padx=5, pady=5)
            tk.Label(results_window, text=x_exprs[i]).grid(row=6+i, column=1, padx=5, pady=5)

        _, ax1 = plt.subplots()

        ffts      = [fft(x_t[i], s_period) for i in range(2)]

        maximum_freq = np.max(
            [ffts[i][0][np.argmax(ffts[i][1])] for i in range(2)]
        )

        for i in range(N):
            ax1.plot(
                ffts[i][0]/maximum_freq,
                ffts[i][1],
                label=f'$x_{i+1}(t)$'
            )

        ax1.set_xlim(0, 3*np.pi)
        ax1.set_xlabel('r [rad/s]')
        ax1.set_ylabel('mag(r) [dB]')
        ax1.legend()

        _, ax2 = plt.subplots()

        for i in range(N):
            ax2.plot(
                t,
                x_t[i],
                label=f'$x_{i+1}(t)$'
            )

        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('x [m]')
        ax2.legend()

        plt.show()


window = HalfCarModelWindow()
window.mainloop()