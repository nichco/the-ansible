import numpy as np
import matplotlib.pyplot as plt

# rosenbrock data
original_rosenbrock = 2.245
localhost_rosenbrock = 3.656
remote_rosenbrock = 5.736

# quartic data mu=1
original_quartic = 1.010
localhost_quartic = 1.045
remote_quartic = 1.595

# quartic data mu=10
original_quartic_mu10 = 3.151
localhost_quartic_mu10 = 5.195
remote_quartic_mu10 = 9.851





categories = ["Original", "Localhost", "Remote"]
problems = ["Rosenbrock (BCD)", "Quartic $\mu=1$ (ALBCD)", "Quartic $\mu=10$ (ALBCD)"]

values = np.array([
	[original_rosenbrock, original_quartic, original_quartic_mu10],
	[localhost_rosenbrock, localhost_quartic, localhost_quartic_mu10],
	[remote_rosenbrock, remote_quartic, remote_quartic_mu10],
])

x = np.arange(len(categories))
width = 0.22

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for index, problem in enumerate(problems):
	offset = (index - 1) * width
	plt.bar(x + offset, values[:, index], width, label=problem)

plt.xticks(x, categories)
plt.ylabel("Time (s)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("timing_data.png", bbox_inches="tight", dpi=500)
plt.show()