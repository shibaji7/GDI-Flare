import matplotlib.pyplot as plt
plt.style.use([
    "/home/shibaji/anaconda3/envs/gdi/lib/python3.11/site-packages/scienceplots/styles/science.mplstyle",
    "/home/shibaji/anaconda3/envs/gdi/lib/python3.11/site-packages/scienceplots/styles/journals/ieee.mplstyle"
])
plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Tahoma",
                    "DejaVu Sans",
                    "Lucida Grande",
                    "Verdana",
                ],
                "font.size": 15,
            }
        )