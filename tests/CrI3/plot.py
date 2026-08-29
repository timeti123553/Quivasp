from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.plotter import BSDOSPlotter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import warnings
import logging
import os


# =========================
# 去掉一些不影响画图的警告
# =========================
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# =========================
# 全局画图风格
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# =========================
# 输入文件
# =========================
vasprun_file = "./vasprun.xml"
kpoints_file = "./KPOINTS"


# =========================
# 检查输入文件是否存在
# =========================
if not os.path.exists(vasprun_file):
    raise FileNotFoundError(
        f"找不到 vasprun.xml：{os.path.abspath(vasprun_file)}"
    )

if not os.path.exists(kpoints_file):
    raise FileNotFoundError(
        f"找不到 KPOINTS：{os.path.abspath(kpoints_file)}\n"
        "请确保 KPOINTS 与 plot.py、vasprun.xml 位于同一目录。"
    )


# =========================
# 读取 DOS
# =========================
print("Reading DOS from vasprun.xml ...")

dos_vasprun = Vasprun(
    vasprun_file,
    parse_dos=True,
    parse_eigen=False,
    parse_projected_eigen=False
)

dos_data = dos_vasprun.complete_dos

print(f"Fermi energy = {dos_vasprun.efermi:.6f} eV")


# =========================
# 读取能带和投影信息
# =========================
print("Reading band structure from vasprun.xml ...")

bs_vasprun = Vasprun(
    vasprun_file,
    parse_dos=False,
    parse_eigen=True,
    parse_projected_eigen=True
)


# =========================
# 获取高对称路径能带
#
# 关键：
# line_mode=True 时，pymatgen 需要 KPOINTS
# 文件来确定 Gamma、M、K 等高对称点标签
# =========================
bs_data = bs_vasprun.get_band_structure(
    kpoints_filename=kpoints_file,
    line_mode=True,
    efermi=dos_vasprun.efermi
)

print("Band structure loaded successfully.")


# =========================
# BSDOSPlotter 设置
# =========================
plotter = BSDOSPlotter(
    bs_projection="elements",
    dos_projection="elements",

    # 费米能级以下显示 5 eV
    vb_energy_range=2,

    # 费米能级以上显示 5 eV
    cb_energy_range=2,

    fixed_cb_energy=False,

    # 能量刻度间隔
    egrid_interval=1,

    font="DejaVu Sans",

    axis_fontsize=20,
    tick_fontsize=16,
    legend_fontsize=15,

    bs_legend="upper right",
    dos_legend="upper right",

    rgb_legend=True,

    fig_size=(12, 8)
)


# =========================
# 作图
# =========================
print("Plotting band structure and DOS ...")

plt_obj = plotter.get_plot(
    bs=bs_data,
    dos=dos_data
)


# =========================
# 获取 figure 和 axes
# =========================
fig = plt.gcf()
axes = fig.get_axes()


# =========================
# 进一步美化每个子图
# =========================
for ax in axes:

    # -------------------------
    # 坐标轴线宽
    # -------------------------
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # -------------------------
    # 主刻度
    # -------------------------
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6,
        width=1.3,
        labelsize=15,
        top=True,
        right=True
    )

    # -------------------------
    # 次刻度
    # -------------------------
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=3,
        width=1.0,
        top=True,
        right=True
    )

    ax.minorticks_on()

    # -------------------------
    # 网格线
    # 如果不喜欢论文图中的网格，
    # 可以把这一段注释掉
    # -------------------------
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.5,
        alpha=0.30
    )

    # -------------------------
    # 调整所有曲线线宽
    # -------------------------
    for line in ax.lines:
        line.set_linewidth(1.8)


# =========================
# 费米能级参考线
#
# BSDOSPlotter 通常已经将 Ef 设为 0 eV
# 这里再加一条 y = 0 虚线
# =========================
if len(axes) > 0:

    bs_ax = axes[0]

    bs_ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8
    )


# =========================
# 标签优化
# =========================
if len(axes) > 0:
    axes[0].set_ylabel(
        r"$E - E_\mathrm{F}$ (eV)",
        fontsize=20
    )


# DOS 子图一般位于右侧
if len(axes) > 1:
    axes[1].set_xlabel(
        "DOS",
        fontsize=18
    )


# =========================
# 布局优化
# =========================
plt.tight_layout()


# =========================
# 保存 PNG
# =========================
png_file = "band_dos.png"

plt.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight",
    transparent=False
)

print(f"Saved: {png_file}")


# =========================
# 保存 PDF
# =========================
pdf_file = "band_dos.pdf"

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

print(f"Saved: {pdf_file}")


# =========================
# 关闭图像
# =========================
plt.close()

print("Done.")

