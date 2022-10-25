import io
import matplotlib.pyplot as plt # 匯入matplotlib 的pyplot 類別，並設定為plt
import xml.etree.ElementTree as ET
from matplotlib.patches import Shadow
from matplotlib.patches import ConnectionPatch
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # GUI 元件
import pandas as pd
import matplotlib.pyplot as plt
#換成中文的字體
# plt.rcParams['font.新細明體'] = ['SimSun'] # 步驟一（替換sans-serif字型）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'

def hat_graph(ax, xlabels, values, group_labels):
    """
    Create a hat graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to plot into.
    xlabels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(xlabels) == N).
    group_labels : list of str
        The group labels displayed in the legend.
    """

    def label_bars(heights, rects):
        """Attach a text label on top of each bar."""
        for height, rect in zip(heights, rects):
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='bottom')

    values = np.asarray(values)
    x = np.arange(values.shape[1])
    ax.set_xticks(x, labels=xlabels)
    spacing = 0.3  # spacing between hat groups
    width = (1 - spacing) / values.shape[0]
    heights0 = values[0]
    for i, (heights, group_label) in enumerate(zip(values, group_labels)):
        style = {'fill': False} if i == 0 else {'edgecolor': 'black'}
        rects = ax.bar(x - spacing/2 + i * width, heights - heights0,
                       width, bottom=heights0, label=group_label, **style)
        label_bars(heights, rects)

#圖一
def char1(listDate1,listy,label1, label2, label3, title, ylabel, xlabel):
    plt.plot(listDate1, listy[0], "y--",label=label1)  # 建立圖表 x軸listDate1 y軸listy[0]
    plt.plot(listDate1, listy[1], "rp--", label=label2)  # 建立圖表 x軸 listDate1y軸listy[1]
    plt.plot(listDate1, listy[2], "cd--",label=label3)  # 建立圖表 x軸 listDate1y軸listy[2]
    plt.legend(loc='upper right')  # 在右上角顯示標籤
    plt.xlabel('七月確診日期')
    plt.ylabel('單日確診人數')

    plt.title('新冠肺炎19 台北,桃園,新竹 確診數量圖表 ')
    plt.savefig("新冠肺炎確診數量圖.jpg")

#圖二
def char2(listDate1, listy1, listy2, listy3, title, ylabel, xlabel,label1,label2,label3):
    x = listDate1
    y1 =listy1
    max = 3
    plt.bar(x, y1,
            alpha=0.5,
            width=1 / max, edgecolor="black",
            linewidth=0.7, label=label1
            )
    x2 = [i + (1 / max) for i in x]
    y2 = listy2
    plt.bar(x2, y2,
            alpha=0.5,
            width=1 / max, edgecolor="white",
            linewidth=0.7, label=label2)
    x3 = [i + (2 / max) for i in x]
    y3 = listy3
    plt.bar(x3, y3,
            alpha=0.5,
            width=1 / max, edgecolor="red",
            linewidth=0.7, label=label3)
    plt.legend(loc='upper right')  # 在右上角顯示標籤
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


#圖九
def char3(listy,listDate1,label1,label2,label3,xlabel,ylabel,title):
    plt.plot(listDate1, listy[0], "go", label=label1)
    plt.plot(listDate1,listy[1], "r_", label=label2)
    plt.plot(listDate1, listy[2], "y-", label=label3)
    plt.legend()  # 自動改變顯示的位置

    plt.title(title)
    plt.ylabel(xlabel)  # 顯示Y 座標的文字
    plt.xlabel(ylabel)  # 顯示Y 座標的文字



def char4(x,y,label1,label2,label3,xlabel,ylabel,title):
    #def charts(x,y,title,xlabel,ylabel):
    #import matplotlib.pyplot as plt
    ### 第一張圖
    # plt.subplot(2, 2, 1, facecolor='y')
    list1=['ro',"go","bo","r-","g-","b-"]
    i=0
    for y2 in y:
        plt.plot(x, y2, list1[i])
        i=i+1

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(rotation=-90, fontsize=8)

def char5(x, y, label1, label2, label3, xlabel, ylabel, title):

    
    ### 第2張圖
    #plt.subplot(2, 2, 2, facecolor='k')

    plt.bar(x, y[0], width=1,
            color="blue",  # 顏色 #ff0000 rgb 三原色
            alpha=0.9,  # 透明度
            edgecolor="white", linewidth=0.7)
    plt.xlabel(title[0])
    plt.ylabel(xlabel[0])
    plt.title(ylabel[0])
    plt.xticks(rotation=-90, fontsize=8)

def char6(x, y, label1, label2, label3, xlabel, ylabel, title):


    #### 第3張圖
    #plt.subplot(2, 2, 3)
    plt.plot(x, y[0], 'b|')
    plt.xlabel(title[0])
    plt.ylabel(xlabel[0])
    plt.title(ylabel[0])
    plt.xticks(rotation=-90, fontsize=8)

def char7(x, y, label1, label2, label3, xlabel, ylabel, title):
    ### 第4張圖
    #plt.subplot(2, 2, 4)
    plt.pie(y[0],  labels=x,
            radius=1,  # 半徑
            center=(4, 4),  # 中心點
            wedgeprops={"linewidth": 1,
                        "edgecolor": "white"},
            frame=True)

    plt.xlabel(title[0])
    plt.ylabel(xlabel[0])
    plt.title(ylabel[0])
    plt.xticks(rotation=-90, fontsize=8)


def char8(x,y,label1,label2,label3,xlabel,ylabel,title):
    #def charts(x,y,title,xlabel,ylabel):
    #import matplotlib.pyplot as plt
    ### 第一張圖
    # plt.subplot(2, 2, 1, facecolor='y')

    plt.plot(x, y[0],"r--")


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(rotation=-90, fontsize=8)


def char9(x, y, label1, label2, label3, xlabel, ylabel, title):
    # def charts(x,y,title,xlabel,ylabel):
    # import matplotlib.pyplot as plt
    ### 第一張圖
    # plt.subplot(2, 2, 1, facecolor='y')

    plt.plot(x, y[0], "r*")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(rotation=-90, fontsize=8)
"""
charts(list民宿店名,list低價位,
       ['烏來區民宿','烏來區民宿','烏來區民宿','烏來區民宿'],
       ['民宿店名','民宿店名','民宿店名','民宿店名'],
       ['低價位','低價位','低價位','低價位'])
"""




#圖9
def subplots_char1(listy1,listDate1,label1,xlabel,ylabel,title,ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # 建立圖表 畫面分割成1個
    ax.plot(listDate1, listy1,"ro")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

#圖9
def subplots_char2(listy1,listDate1,label1,xlabel,ylabel,title,ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # 建立圖表 畫面分割成1個
    ax.plot(listDate1, listy1,"b--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
#圖三
def subplots_char3(listDate1, listy1, listy2, listy3, title, ylabel, xlabel,label1,label2,label3,ax=None):
       xlabels = listDate1
       Taipei = np.array(listy1)
       Taoyun = np.array(listy2)
       Hsinchu = np.array(listy3)
       if ax is None:
            fig, ax = plt.subplots()
       hat_graph(ax, xlabels, [Taipei, Taoyun, Hsinchu], [label1,label2, label3])

       # Add some text for labels, title and custom x-axis tick labels, etc.
       ax.set_xlabel(xlabel)
       ax.set_ylabel(ylabel)
       ax.set_ylim(0, 3000)
       ax.set_title(title)
       ax.legend()
#圖四
def subplots_char4(listDate1, listy, title,list1Label,ax=None):

    vegetables =list1Label
    farmers = listDate1

    harvest = np.array([listy[0],listy[1],listy[2]])

    ax輸入為空=None
    if ax is None:
        fig, ax = plt.subplots()
        ax輸入為空=True
    else:
        ax輸入為空=False
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="r")

    ax.set_title(title)
    if ax輸入為空==True:
        fig.tight_layout()
#圖五
def subplots_char5(list1Label,listy1,listy2,listy3,ax=None):

    if ax is not None:
        ax.plot( listy1)
    else:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        labels =list1Label
        fracs = [listy1[0],listy2[0], listy3[0]]

        explode = (0, 0.05, 0)

        # We want to draw the shadow for each pie but we will not use "shadow"
        # option as it doesn't save the references to the shadow patches.
        pies = ax.pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%')

        for w in pies[0]:
            # set the id with the label.
            w.set_gid(w.get_label())

            # we don't want to draw the edge of the pie
            w.set_edgecolor("none")

        for w in pies[0]:
            # create shadow patch
            s = Shadow(w, -0.01, -0.01)
            s.set_gid(w.get_gid() + "_shadow")
            s.set_zorder(w.get_zorder() - 0.1)
            ax.add_patch(s)

        # save
        f = io.BytesIO()
        plt.savefig(f, format="svg")

        # Filter definition for shadow using a gaussian blur and lighting effect.
        # The lighting filter is copied from http://www.w3.org/TR/SVG/filters.html

        # I tested it with Inkscape and Firefox3. "Gaussian blur" is supported
        # in both, but the lighting effect only in Inkscape. Also note
        # that, Inkscape's exporting also may not support it.

        filter_def = """
          <defs xmlns='http://www.w3.org/2000/svg'
                xmlns:xlink='http://www.w3.org/1999/xlink'>
            <filter id='dropshadow' height='1.2' width='1.2'>
              <feGaussianBlur result='blur' stdDeviation='2'/>
            </filter>
    
            <filter id='MyFilter' filterUnits='objectBoundingBox'
                    x='0' y='0' width='1' height='1'>
              <feGaussianBlur in='SourceAlpha' stdDeviation='4%' result='blur'/>
              <feOffset in='blur' dx='4%' dy='4%' result='offsetBlur'/>
              <feSpecularLighting in='blur' surfaceScale='5' specularConstant='.75'
                   specularExponent='20' lighting-color='#bbbbbb' result='specOut'>
                <fePointLight x='-5000%' y='-10000%' z='20000%'/>
              </feSpecularLighting>
              <feComposite in='specOut' in2='SourceAlpha'
                           operator='in' result='specOut'/>
              <feComposite in='SourceGraphic' in2='specOut' operator='arithmetic'
            k1='0' k2='1' k3='1' k4='0'/>
            </filter>
          </defs>
        """

        tree, xmlid = ET.XMLID(f.getvalue())

        # insert the filter definition in the svg dom tree.
        tree.insert(0, ET.XML(filter_def))

        for i, pie_name in enumerate(labels):
            pie = xmlid[pie_name]
            pie.set("filter", 'url(#MyFilter)')

            shadow = xmlid[pie_name + "_shadow"]
            shadow.set("filter", 'url(#dropshadow)')

        fn = "svg_filter_pie.svg"
        print(f"Saving '{fn}'")
        ET.ElementTree(tree).write(fn)
#圖六
def subplots_char6(list1Label,listy1,listy2,listy3,ax=None):

    # Some data
    labels = list1Label
    fracs = [listy1[0],listy2[0], listy3[0]]

    # Make figure and axes
    if ax is None:
        fig, axs = plt.subplots(2, 2)
        ax1=axs[0,0]
        ax2=axs[0,1]
        ax3=axs[1,0]
        ax4=axs[1,1]
    else:
        ax1=ax
        ax2=None
        ax3=None
        ax4=None


    # A standard pie plot
    ax1.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True)

    # Shift the second slice using explode
    if ax2 is not None:
        ax2.pie(fracs, labels=labels, autopct='%.0f%%', shadow=True,
                  explode=(0, 0.1, 0))

    # Adapt radius and text size for a smaller pie
    if ax3 is not None:
        patches, texts, autotexts = ax3.pie(fracs, labels=labels,
                                              autopct='%.0f%%',
                                              textprops={'size': 'smaller'},
                                              shadow=True, radius=0.5)
        # Make percent texts even smaller
        plt.setp(autotexts, size='x-small')
        autotexts[0].set_color('white')


    # Use a smaller explode and turn of the shadow for better visibility
    if ax4 is not None:
        patches, texts, autotexts = ax4.pie(fracs, labels=labels,
                                              autopct='%.0f%%',
                                              textprops={'size': 'smaller'},
                                              shadow=False, radius=0.5,
                                              explode=(0, 0.05, 0))
        plt.setp(autotexts, size='x-small')
        autotexts[0].set_color('white')
#圖七
def subplots_char7(list1Label,listy1,listy2,listy3,ax=None):
    # make figure and assign axis objects
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
        fig.subplots_adjust(wspace=0)
    else:
        ax1=ax
        ax2=None

    # pie chart parameters
    overall_ratios = [listy1[0], listy2[0], listy3[0]]
    labels =list1Label
    explode = [0.1, 0, 0]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * overall_ratios[0]
    wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                         labels=labels, explode=explode)

    # bar chart parameters
    age_ratios = [.33, .54, .07, .06]
    age_labels = ['Under 35', '35-49', '50-65', 'Over 65']
    bottom = 1
    width = .2

    # Adding from the top matches the legend.
    if ax2 is not None:
        for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
            bottom -= height
            bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                         alpha=0.1 + 0.25 * j)
            ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

        ax2.set_title('Age of approvers')
        ax2.legend()
        ax2.axis('off')
        ax2.set_xlim(- 2.5 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        theta1, theta2 = wedges[0].theta1, wedges[0].theta2
        center, r = wedges[0].center, wedges[0].r
        bar_height = sum(age_ratios)

        # draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                              xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        con.set_linewidth(4)
        ax2.add_artist(con)

        # draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                              xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        ax2.add_artist(con)
        con.set_linewidth(4)
#圖八
def subplots_char8(listy1,listDate1,label1,xlabel,ylabel,title,ax=None):
    if ax is None:
       fig, ax = plt.subplots()  # 建立圖表 畫面分割成1個

    x = listDate1
    y1 = listy1
    ax.bar(x, y1,
            alpha=0.5,
            width=1, edgecolor="black",
            linewidth=0.7, label=label1
            )

    ax.legend(loc='upper right')  # 在右上角顯示標籤

    #ax.xlabel(xlabel)
    #ax.ylabel(ylabel)

    #ax.title(title)



#圖9
def subplots_char9(listy1,listDate1,label1,xlabel,ylabel,title,ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # 建立圖表 畫面分割成1個
    ax.plot(listDate1, listy1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


#九宮格
def NineCharts(list1Label,listDate1,listy, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3):

    plt.subplot(3,3,1) #, facecolor='y')
    char1(listDate1,listy, label1, label2, label3, title, ylabel, xlabel)
    plt.subplot(3,3,2)
    char2(listDate1, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3)
    plt.subplot(3,3,3)
    char3(listy, listDate1, label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 4)
    char4(listDate1,listy,  label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 5)
    char5(listDate1,listy,  label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 6)
    char6(listDate1,listy,  label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 7)
    char7(listDate1,listy,  label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 8)
    char8(listDate1,listy,  label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 9)
    char9(listDate1,listy,  label1, label2, label3, xlabel, ylabel, title)

    """
    plt.subplot(3, 3, 4)
    char4(listDate1, listy, title, list1Label)
    
    plt.subplot(3, 3, 5)
    char5(list1Label, listy1, listy2, listy3)
    plt.subplot(3, 3, 6)
    char6(list1Label, listy1, listy2, listy3)
    plt.subplot(3, 3, 7)
    char7(list1Label, listy1, listy2, listy3)
    plt.subplot(3, 3, 8)
    char8(listy1, listDate1, label1, xlabel, ylabel, title)
    plt.subplot(3, 3, 9)
    char9(listy, listDate1, label1, label2, label3, xlabel, ylabel, title)
    """


# 九宮格
def subplots_NineCharts(list1Label,listDate1,listy, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3):

    fig, ax = plt.subplots(3,3)  # 建立圖表 畫面分割成1個
    subplots_char1(listy1, listDate1, label1, xlabel, ylabel, title,ax[0,0])
    subplots_char2(listy1,listDate1,label1,xlabel,ylabel,title,ax[0,1])
    subplots_char3(listDate1, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3,ax[0,2])
    subplots_char4(listDate1, listy, title,list1Label,ax[1,0])
    subplots_char5(list1Label,listy1,listy2,listy3,ax[1,1])
    subplots_char6(list1Label,listy1,listy2,listy3,ax[1,2])
    subplots_char7(list1Label,listy1,listy2,listy3,ax[2,0])
    subplots_char8(listy1,listDate1,label1,xlabel,ylabel,title,ax[2,1])
    subplots_char9(listy1,listDate1,label1,xlabel,ylabel,title,ax[2,2])



def pandas_取得裡面的種類(df,columeName):
    list1 = df[columeName].unique()
    return list1

from matplotlib.font_manager import FontProperties  # 中文字體


def matplot_中文字():

    # 換成中文的字體
    # plt.rcParams['font.新細明體'] = ['SimSun'] # 步驟一（替換sans-serif字型）
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）
