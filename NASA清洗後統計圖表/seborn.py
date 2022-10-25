import seaborn as sns    # pip install seaborn    # 匯入函示庫 seaborn 換名 sns
import pandas as pd# 匯入函示庫 pandas 換名 pd
import matplotlib as mpl # matplotlib
mpl.use("TKAgg")
import matplotlib.pyplot as plt                   # 匯入函示庫 matplotlib.pyplot 換名 plt
import sys


from matplotlib.font_manager import FontProperties

if sys.platform.startswith("linux"):  # could be "linux", "linux2", "linux3", ...
    print("linux")  # linux
elif sys.platform == "darwin":  # MAC OS X
    from matplotlib.font_manager import FontProperties      # 中文字體
    plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font="Arial Unicode MS") #"DFKai-SB")
elif sys.platform == "win32":
    # Windows (either 32-bit or 64-bit)
    sns.set(font="sans-serif") #"DFKai-SB")        #<--- 注意： 位置
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 換成中文的字體
    plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）
               # 中文字體



sns.set_theme(style="whitegrid")                  # 設定主題

df=pd.read_excel("NASA(清洗後).xlsx") #讀取清洗後資料

print(df.head())

#kdeplot
sns.kdeplot(
   data=df, x="relative_velocity", hue="hazardous",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)
plt.savefig("kdeplot.png")
plt.show()

#displot
sns.displot(data=df, x="relative_velocity", y="miss_distance", hue="hazardous", kind="kde")

plt.savefig("displot.png")
plt.show()

#relplot
sns.relplot(data=df, x="relative_velocity", y="miss_distance", hue="hazardous")
plt.savefig("relplot.png")
plt.show()

#pairplot
sns.pairplot(df, hue="hazardous")
plt.savefig("pairplot.png")
plt.show()