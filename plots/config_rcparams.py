# -*- coding: utf-8 -*-
# @author: Vincent Thibeault and rcParams shared by Frédéric Dion

import matplotlib.pyplot as plt
import matplotlib.colors

# Configuration of LaTeX environment for plots generated with matplotlib.pyplot
# General unit of measure "points"

""" Colors """
dark_grey = '#404040'                       # RGB: 48, 48, 48     dark grey
complete_grey = "#252525"                   # RGB: 37, 37, 37     dark grey 2
reduced_grey = "#969696"                    # RGB: 150, 150, 150  light grey
total_color = "#525252"                     # RGB: 82, 82, 82     grey
first_community_color = "#2171b5"           # RGB: 33, 113, 181   blue
second_community_color = "#f16913"          # RGB: 241, 105, 19   orange
third_community_color = "#238b45"           # RGB: 35, 139, 69    green
fourth_community_color = "#6a51a3"          # RGB: 106, 81, 163   purple
reduced_first_community_color = "#9ecae1"   # RGB: 158, 202, 225  light blue
reduced_second_community_color = "#fdd0a2"  # RGB: 253, 208, 162  light orange
reduced_third_community_color = "#a1d99b"   # RGB: 161, 217, 155  light green
reduced_fourth_community_color = "#9e9ac8"  # RGB: 158, 154, 200  light purple


""" Font and text """
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.style'] = "normal"
plt.rcParams['font.variant'] = "normal"
plt.rcParams['font.weight'] = "medium"
plt.rcParams['font.stretch'] = "normal"
plt.rcParams['font.size'] = 12
# plt.rcParams['font.serif'] = DejaVu Serif, Bitstream Vera Serif,
#                              New Century Schoolbook, Century Schoolbook L,
#                              Utopia, ITC Bookman, Bookman,
#                              Nimbus Roman No9 L, Times New Roman,
#                              Times, Palatino, Charter, serif
# plt.rcParams['font.sans-serif'] = DejaVu Sans, Bitstream Vera Sans,
#                                   Lucida Grande, Verdana, Geneva, Lucid,
#                                   Arial, Helvetica, Avant Garde, sans-serif
# plt.rcParams['font.cursive'] = Apple Chancery, Textile, Zapf Chancery,
#                                Sand, Script MT, Felipa, cursive
# plt.rcParams['font.fantasy'] = Comic Sans MS, Chicago, Charcoal, Impact,
#                                Western, Humor Sans, xkcd, fantasy
# plt.rcParams['font.monospace'] = DejaVu Sans Mono, Bitstream Vera Sans Mono,
#                                  Andale Mono, Nimbus Mono L, Courier New,
#                                  Courier, Fixed, Terminal, monospace
# font = {'family': 'serif', 'serif': ['Times New Roman']}
# plt.rc('font', **font)
# plt.rc('text', usetex=True)
# plt.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{lmodern}\usepackage{sfmath}')
plt.rcParams['text.color'] = dark_grey
plt.rcParams['axes.labelcolor'] = dark_grey
plt.rcParams['xtick.color'] = dark_grey
plt.rcParams['ytick.color'] = dark_grey
inset_fontsize = 9
fontsize_legend = 12
inset_labelsize = 9
linewidth = 2

""" Axes """
plt.rcParams['axes.facecolor'] = "white"     # axes background color
plt.rcParams['axes.edgecolor'] = dark_grey   # axes edge color
plt.rcParams['axes.linewidth'] = 1.1         # edge linewidth
plt.rcParams['axes.grid'] = False            # display grid or not
plt.rcParams['axes.titlesize'] = "large"     # fontsize of the axes title
plt.rcParams['axes.titlepad'] = 6            # pad between axes and title
plt.rcParams['axes.labelsize'] = 12          # fontsize of the x any y labels
plt.rcParams['axes.labelpad'] = 4            # space between label and axis
plt.rcParams['axes.labelweight'] = "normal"  # weight of the x and y labels
plt.rcParams['axes.labelcolor'] = dark_grey  # label color
plt.rcParams['axes.axisbelow'] = 'line'      # draw axis gridlines+ticks below
plt.rcParams['axes.spines.top'] = False      # draw top spine of the axis
plt.rcParams['axes.spines.bottom'] = True    # draw bottom spine of the axis
plt.rcParams['axes.spines.left'] = True      # draw left spine of the axis
plt.rcParams['axes.spines.right'] = False    # draw right spine of the axis

""" xticks """
plt.rcParams['xtick.top'] = True           # draw ticks on the top side
plt.rcParams['xtick.bottom'] = True        # draw ticks on the bottom side
plt.rcParams['xtick.major.size'] = 5.0     # major tick size
plt.rcParams['xtick.minor.size'] = 5.0     # minor tick size
plt.rcParams['xtick.major.width'] = 1.1    # major tick width
plt.rcParams['xtick.minor.width'] = 1.1    # minor tick width
plt.rcParams['xtick.major.pad'] = 3.5      # distance to major tick label
plt.rcParams['xtick.minor.pad'] = 3.5      # distance to the minor tick label
plt.rcParams['xtick.color'] = dark_grey    # color of the tick labels
plt.rcParams['xtick.labelsize'] = 12       # fontsize of the tick labels
plt.rcParams['xtick.direction'] = 'out'    # direction: in, out, or inout
plt.rcParams['xtick.major.top'] = False    # draw x axis top major ticks
plt.rcParams['xtick.major.bottom'] = True  # draw x axis bottom major ticks
plt.rcParams['xtick.minor.top'] = True     # draw x axis top minor ticks
plt.rcParams['xtick.minor.bottom'] = True  # draw x axis bottom minor ticks

""" yticks """
plt.rcParams['ytick.left'] = True            # draw ticks on the left side
plt.rcParams['ytick.right'] = True           # draw ticks on the right side
plt.rcParams['ytick.major.size'] = 5.0       # major tick size in points
plt.rcParams['ytick.minor.size'] = 5.0       # minor tick size in points
plt.rcParams['ytick.major.width'] = 1.1      # major tick width in points
plt.rcParams['ytick.minor.width'] = 1.1      # minor tick width in points
plt.rcParams['ytick.major.pad'] = 3.5        # distance to major tick label
plt.rcParams['ytick.minor.pad'] = 3.5        # distance to the minor tick label
plt.rcParams['ytick.color'] = dark_grey      # color of the tick labels
plt.rcParams['ytick.labelsize'] = 12         # fontsize of the tick labels
plt.rcParams['ytick.direction'] = "out"      # direction: in, out, or inout
plt.rcParams['ytick.minor.visible'] = False  # visibility minor ticks on y-axis
plt.rcParams['ytick.major.left'] = True      # draw y axis left major ticks
plt.rcParams['ytick.major.right'] = False    # draw y axis right major ticks
plt.rcParams['ytick.minor.left'] = True      # draw y axis left minor ticks
plt.rcParams['ytick.minor.right'] = True     # draw y axis right minor ticks

""" Grids """
plt.rcParams['grid.linestyle'] = "-"  # solid
plt.rcParams['grid.linewidth'] = 0.8  # in points
plt.rcParams['grid.alpha'] = 1.0      # transparency, between 0.0 and 1.0

""" Legend """
plt.rcParams['legend.loc'] = "best"           # "upper right"
plt.rcParams['legend.frameon'] = False         # if True, draw the legend on a background patch
plt.rcParams['legend.framealpha'] = 0.9       # legend patch transparency
plt.rcParams['legend.facecolor'] = "inherit"  # inherit from axes.facecolor; or color spec
plt.rcParams['legend.edgecolor'] = dark_grey  # background patch boundary color
plt.rcParams['legend.fancybox'] = True        # if True, use a rounded box for the
plt.rcParams['legend.shadow'] = False         # if True, give background a shadow effect
plt.rcParams['legend.numpoints'] = 1          # the number of marker points in the legend line
plt.rcParams['legend.scatterpoints'] = 1      # number of scatter points
plt.rcParams['legend.markerscale'] = 1.0      # the relative size of legend markers vs. original
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.borderpad'] = 0.4        # border whitespace
plt.rcParams['legend.labelspacing'] = 0.5     # the vertical space between the legend entries
plt.rcParams['legend.handlelength'] = 2.0     # the length of the legend lines
plt.rcParams['legend.handleheight'] = 0.7     # the height of the legend handle
plt.rcParams['legend.handletextpad'] = 0.8    # the space between the legend line and legend text
plt.rcParams['legend.borderaxespad'] = 0.5    # the border between the axes and legend edge
plt.rcParams['legend.columnspacing'] = 2.0    # column separation

""" Figure """
plt.rcParams['figure.titlesize'] = "large"      # size of the figure title (Figure.suptitle())
plt.rcParams['figure.titleweight'] = "normal"   # weight of the figure title
plt.rcParams['figure.figsize'] = [8, 6]         # figure size in inches
plt.rcParams['figure.dpi'] = 150                # figure dots per inch
plt.rcParams['figure.facecolor'] = "white"      # figure facecolor; 0.75 is scalar gray
plt.rcParams['figure.edgecolor'] = "white"      # figure edgecolor
plt.rcParams['figure.autolayout'] = True        # When True, automatically adjust subplot
plt.rcParams["patch.force_edgecolor"] = True

""" Patch """
plt.rcParams["patch.force_edgecolor"] = False

""" Colormaps """
cdict = {
    'red':   ((0, 255/255, 255/255),
              (0.4, 253 / 255, 253 / 255),
              (0.6, 253 / 255, 253 / 255),
              (0.8, 253 / 255, 253 / 255),
              (1.0, 241 / 255, 241 / 255),),
    'green': ((0, 245/255, 245/255),
              (0.4, 208 / 255, 208 / 255),
              (0.6, 174 / 255, 174 / 255),
              (0.8, 141 / 255, 141 / 255),
              (1.0, 105 / 255, 105 / 255),),
    'blue':  ((0, 235/255, 235/255),
              (0.4, 162 / 255, 162 / 255),
              (0.6, 107 / 255, 107 / 255),
              (0.8, 60 / 255,  60 / 255),
              (1.0, 19 / 255, 19 / 255))
}
cm = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

cdict2 = {
    'red':   ((0,   241 / 255, 241 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0,  33 / 255,  33 / 255),),
    'green': ((0,   105 / 255, 105 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 113 / 255, 113 / 255),),
    'blue':  ((0,    19 / 255,  19 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 181 / 255, 181 / 255))
}
cm2 = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict2, 1024)

cdict3 = {
    'red':   ((0,   158 / 255, 158 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 253 / 255, 253 / 255),),
    'green': ((0,   202 / 255, 202 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 208 / 255, 208 / 255),),
    'blue':  ((0,   225 / 255, 225 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 162 / 255, 162 / 255))
}
cm3 = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict3, 1024)

cdict4 = {
    'red':   ((0,   255 / 255, 255 / 255),
              (0.1, 158 / 255, 158 / 255),
              (1.0, 253 / 255, 253 / 255),),
    'green': ((0,   255 / 255, 255 / 255 ),
              (0.1, 202 / 255, 202 / 255),
              (1.0, 208 / 255, 208 / 255),),
    'blue':  ((0,   255 / 255, 255 / 255),
              (0.1, 225 / 255, 225 / 255),
              (1.0, 162 / 255, 162 / 255))
}
cm4 = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict4, 1024)
