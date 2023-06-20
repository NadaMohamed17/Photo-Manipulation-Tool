import tkinter as tk
import tkinter.font as font
from tkinter import *
from tkinter import filedialog

import matplotlib
import matplotlib.image as img
import numpy as np
from PIL import Image, ImageTk

matplotlib.use( "TkAgg" )
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure
import funcs

global imageDisplay1
global second_image
import cv2 as cv

main_page = tk.Tk()
main_page.configure( bg='#20272d' )
main_page.geometry( "1920x1080" )  # Size of the window
winheight = 1080
main_page.title( 'DIP' )
my_font1 = ('times', 18, 'bold')
labelfont = ('Helvetica', 11)
systembtncolor = '#03DAC6'
framecolor = '#2d363f'
labelfontcolor = '#E6E6E6'
buttoncolor = '#5d666f'
togglebtncolor = '#2d363f'
# variable messagebox
VARIABLELABEL = 'Enter Number'
myFont = font.Font( size=14 )
histogram_icon = PhotoImage( file="icons/histogram.png" )
arth_icon = PhotoImage( file="icons/arth.png" )
bit_icon = PhotoImage( file="icons/binary.png" )
filters_icon = PhotoImage( file="icons/filters.png" )
contrast_icon = PhotoImage( file="icons/contrast.png" )
threshold_icon = PhotoImage( file="icons/subtract.png" )
restore_icon = PhotoImage( file="icons/restore.png" )

uploadFrame = Frame( main_page, width=40, height=1080, bg=framecolor, highlightthickness=0 )
# uploadFrame.config( highlightbackground='#101010' )
uploadFrame.place( x=0, y=0 )



def choice(e):
    value = int( e.get() )
    pop.destroy()
    printlabel( funcs.arthadd( funcs.setappdata, value ) )


def addpopup():
    global pop
    pop = Toplevel( main_page )
    pop.resizable( False, False )
    pop.title( VARIABLELABEL )
    pop.geometry( '250x150' )
    pop.config( bg=framecolor )
    Label( pop, text=VARIABLELABEL, fg=labelfontcolor, bg=framecolor ).pack( pady=15 )
    popvariable = Entry( pop )

    button = tk.Button( pop, text='Proceed', width=20, bg='#BB86FC', fg='#000000', borderwidth=3,
                        command=lambda: choice( popvariable ) )

    button.place( x=45, y=100 )
    popvariable.place( x=60, y=50 )


# multiply variable
def multiplychoice(e):
    value = int( e.get() )
    multiplypop.destroy()
    printlabel( funcs.arthmultiply( funcs.setappdata, value ) )


def multiplypopup():
    global multiplypop
    multiplypop = Toplevel( main_page )
    multiplypop.resizable( False, False )
    multiplypop.title( VARIABLELABEL )
    multiplypop.geometry( '250x150' )
    multiplypop.config( bg=framecolor )
    Label( multiplypop, text=VARIABLELABEL, fg=labelfontcolor, bg=framecolor ).pack( pady=15 )
    multiplypopvariable = Entry( multiplypop )

    button = tk.Button( multiplypop, text='Proceed', width=20, bg='#BB86FC', fg='#000000', borderwidth=3,
                        command=lambda: multiplychoice( multiplypopvariable ) )

    button.place( x=45, y=100 )
    multiplypopvariable.place( x=60, y=50 )


# division variable
def divisionchoice(e):
    value = int( e.get() )
    divisionpop.destroy()
    printlabel( funcs.arthdivison( funcs.setappdata, value ) )


def divisionpopup():
    global divisionpop
    divisionpop = Toplevel( main_page )
    divisionpop.resizable( False, False )
    divisionpop.title( VARIABLELABEL )
    divisionpop.geometry( '250x150' )
    divisionpop.config( bg=framecolor )
    Label( divisionpop, text=VARIABLELABEL, fg=labelfontcolor, bg=framecolor ).pack( pady=15 )
    divisionpopvariable = Entry( divisionpop )

    button = tk.Button( divisionpop, text='Proceed', width=20, bg='#BB86FC', fg='#000000', borderwidth=3,
                        command=lambda: divisionchoice( divisionpopvariable ) )

    button.place( x=45, y=100 )
    divisionpopvariable.place( x=60, y=50 )


# subtract variable
def subtractchoice(e):
    value = int( e.get() )
    subtractpop.destroy()
    printlabel( funcs.arthsubtract( funcs.setappdata, value ) )


def subtractpopup():
    global subtractpop
    subtractpop = Toplevel( main_page )
    subtractpop.resizable( False, False )
    subtractpop.title( VARIABLELABEL )
    subtractpop.geometry( '250x150' )
    subtractpop.config( bg=framecolor )
    Label( subtractpop, text=VARIABLELABEL, fg=labelfontcolor, bg=framecolor ).pack( pady=15 )
    subtractpopvariable = Entry( subtractpop )

    button = tk.Button( subtractpop, text='Proceed', width=20, bg='#BB86FC', fg='#000000', borderwidth=3,
                        command=lambda: subtractchoice( subtractpopvariable ) )

    button.place( x=45, y=100 )
    subtractpopvariable.place( x=60, y=50 )


# powerlaw variable
def powerlawchoice(e):
    value = int( e.get() )
    powerlawpop.destroy()
    printlabel( funcs.Power_law_transformation( cvim, value ) )


def powerlawpopup():
    global powerlawpop
    powerlawpop = Toplevel( main_page )
    powerlawpop.resizable( False, False )
    powerlawpop.title( VARIABLELABEL )
    powerlawpop.geometry( '250x150' )
    powerlawpop.config( bg=framecolor )
    Label( powerlawpop, text=VARIABLELABEL, fg=labelfontcolor, bg=framecolor ).pack( pady=15 )
    powerpopvariable = Entry( powerlawpop )

    button = tk.Button( powerlawpop, text='Proceed', width=20, bg='#BB86FC', fg='#000000', borderwidth=3,
                        command=lambda: powerlawchoice( powerpopvariable ) )

    button.place( x=45, y=100 )
    powerpopvariable.place( x=60, y=50 )


# graylevel variable
def graylevelchoiceA1(graylevellow, graylevelhigh):
    low = int( graylevellow.get() )
    high = int( graylevelhigh.get() )
    graylevelpopA1.destroy()
    printlabel( funcs.GraylevelslicingA1( cvim, high, low ) )


def graylevelpopupA1():
    global graylevelpopA1
    graylevelpopA1 = Toplevel( main_page )
    graylevelpopA1.resizable( False, False )
    graylevelpopA1.title( VARIABLELABEL )
    graylevelpopA1.geometry( '250x150' )
    graylevelpopA1.config( bg=framecolor )
    Label( graylevelpopA1, text=VARIABLELABEL, fg=labelfontcolor, bg=framecolor ).pack()
    Label( graylevelpopA1, text='High', fg=labelfontcolor, bg=framecolor ).place( x=10, y=60 )
    Label( graylevelpopA1, text='Low', fg=labelfontcolor, bg=framecolor ).place( x=10, y=30 )
    graylevellow = Entry( graylevelpopA1 )
    graylevelhigh = Entry( graylevelpopA1 )

    button = tk.Button( graylevelpopA1, text='Proceed', width=20, bg='#BB86FC', fg='#000000', borderwidth=3,
                        command=lambda: graylevelchoiceA1( graylevellow, graylevelhigh ) )

    button.place( x=45, y=100 )
    graylevelhigh.place( x=60, y=60 )
    graylevellow.place( x=60, y=30 )


def graylevelchoiceA2(graylevellow, graylevelhigh):
    low = int( graylevellow.get() )
    high = int( graylevelhigh.get() )
    graylevelA2pop.destroy()
    printlabel( funcs.GraylevelslicingA2( cvim, high, low ) )


def graylevelpopupA2():
    global graylevelA2pop
    graylevelA2pop = Toplevel( main_page )
    graylevelA2pop.resizable( False, False )
    graylevelA2pop.title( VARIABLELABEL )
    graylevelA2pop.geometry( '250x150' )
    graylevelA2pop.config( bg=framecolor )
    Label( graylevelA2pop, text=VARIABLELABEL, fg=labelfontcolor, bg=framecolor ).pack()
    Label( graylevelA2pop, text='High', fg=labelfontcolor, bg=framecolor ).place( x=10, y=60 )
    Label( graylevelA2pop, text='Low', fg=labelfontcolor, bg=framecolor ).place( x=10, y=30 )
    graylevellow = Entry( graylevelA2pop )
    graylevelhigh = Entry( graylevelA2pop )

    button = tk.Button( graylevelA2pop, text='Proceed', width=20, bg='#BB86FC', fg='#000000', borderwidth=3,
                        command=lambda: graylevelchoiceA2( graylevellow, graylevelhigh ) )

    button.place( x=45, y=100 )
    graylevelhigh.place( x=60, y=60 )
    graylevellow.place( x=60, y=30 )


# threshold variable
def thresholdchoice(k):
    kvar = int( k.get() )
    thresholdpop.destroy()
    printcanvas( funcs.Thresholding( cvim, kvar ) )


def thresholdpopup():
    global thresholdpop
    thresholdpop = Toplevel( main_page )
    thresholdpop.resizable( False, False )
    thresholdpop.title( VARIABLELABEL )
    thresholdpop.geometry( '250x150' )
    thresholdpop.config( bg=framecolor )
    Label( thresholdpop, text=VARIABLELABEL, fg=labelfontcolor, bg=framecolor ).pack( pady=15 )
    k = Entry( thresholdpop )

    button = tk.Button( thresholdpop, text='Proceed', width=20, bg='#BB86FC', fg='#000000', borderwidth=3,
                        command=lambda: thresholdchoice( k ) )

    button.place( x=45, y=100 )
    k.place( x=60, y=50 )


def selected():
    global image_path, imageprosses
    global image1
    global img1
    global imgpil
    global cvim
    global f_types, imagerestore
    f_types = [('Jpg Files', '*.jpg'), ('Jpeg Files', '*.jpeg'), ('PNG Files', '*.png'), ('SVG Files', '*.svg'),
               ('WebP Files', '*.webp')]
    imageDisplay1 = filedialog.askopenfilename( filetypes=f_types )
    imgpil = Image.open( imageDisplay1 )
    img_resized = imgpil.resize( (255, 255) )
    img1 = ImageTk.PhotoImage( img_resized )
    image_display = tk.Label( main_page, image=img1, borderwidth=0 )
    image_display.image = img1
    image_display.place( x=1100, y=300 )
    image1 = img.imread( imageDisplay1 )
    cvim = cv.imread( imageDisplay1, 0 )
    histogram( img1 )
    funcs.setappdata(image1)


arthematic_btn = tk.Button( uploadFrame, image=arth_icon, width=24, height=24, bg=framecolor, fg='#808080',
                            borderwidth=0,
                            command=lambda: onclickarth() )
histo_btn = tk.Button( uploadFrame, image=histogram_icon, width=24, height=24, bg=framecolor, fg='#808080',
                       borderwidth=0, command=lambda: onclickhisto() )
contrast_btn = tk.Button( uploadFrame, image=contrast_icon, width=24, height=24, bg=framecolor, fg='#808080',
                          borderwidth=0, command=lambda: onclickfreq() )
threshold_btn = tk.Button( uploadFrame, image=threshold_icon, width=32, height=32, bg=framecolor, fg='#808080',
                           borderwidth=0, command=lambda: onclickbit() )
filters_btn = tk.Button( uploadFrame, image=filters_icon, width=24, height=24, bg=framecolor, fg='#808080',
                         borderwidth=0, command=lambda: onclickfilter() )
bit_btn = tk.Button( uploadFrame, image=bit_icon, width=24, height=24, bg=framecolor, fg='#808080', borderwidth=0,
                     command=lambda: onclicklinear() )

arthematic_btn.place( x=5, y=20 )
histo_btn.place( x=5, y=60 )
contrast_btn.place( x=5, y=100 )
threshold_btn.place( x=5, y=140 )
filters_btn.place( x=5, y=180 )
bit_btn.place( x=5, y=220 )
uploadfirst = tk.Button( main_page, text='Upload Image', width=20, bg=systembtncolor, fg='#000000',
                         borderwidth=3,
                         command=lambda: selected() )
uploadfirst.place( x=1160, y=650 )
canvas2 = Canvas( main_page, width="600", height="600", bd=2, highlightthickness=0 )
canvas2.configure( bg='#20272d' )
canvas2.place( x=400, y=30 )


def onclickarth():
    global arthematic_toggle
    bgcolor = "#1e252b";
    arthematic_toggle = Frame( main_page, width=145, height=1000, bg=bgcolor )
    arthematic_toggle.place( x=40, y=0 )
    add_button = tk.Button( arthematic_toggle, width=20, text="Add", height=2, highlightthickness=0, borderwidth=0,
                            background=togglebtncolor, foreground='#E1D9D1',
                            command=lambda: addpopup() ).place( x=0, y=0 )
    subtract_button = tk.Button( arthematic_toggle, width=20, text="subtract", height=2, highlightthickness=0,
                                 borderwidth=0,
                                 background=togglebtncolor, foreground='#E1D9D1',
                                 command=lambda: subtractpopup() ).place( x=0, y=37 )
    multiply_button = tk.Button( arthematic_toggle, width=20, text="multiply", height=2, highlightthickness=0,
                                 borderwidth=0,
                                 background=togglebtncolor, foreground='#E1D9D1',
                                 command=lambda: multiplypopup() ).place( x=0, y=74 )
    division_button = tk.Button( arthematic_toggle, width=20, text="division", height=2, highlightthickness=0,
                                 borderwidth=0,
                                 background=togglebtncolor, foreground='#E1D9D1',
                                 command=lambda: divisionpopup() ).place( x=0, y=111 )
    logand = tk.Button( arthematic_toggle, width=20, text="logic and", height=2, highlightthickness=0,
                        borderwidth=0,
                        background=togglebtncolor, foreground='#E1D9D1',
                        command=lambda: printlabel( funcs.logic_and( cvim ) ) ).place( x=0, y=148 )
    logor = tk.Button( arthematic_toggle, width=20, text="logic or", height=2, highlightthickness=0,
                       borderwidth=0,
                       background=togglebtncolor, foreground='#E1D9D1',
                       command=lambda: printlabel( funcs.logic_or( cvim ) ) ).place( x=0, y=185 )


def onclickbit():
    bgcolor = "#1e252b";
    bit_toggle = Frame( main_page, width=145, height=1000, bg=bgcolor )
    bit_toggle.place( x=40, y=0 )
    contrast_stretching = tk.Button( bit_toggle, width=20, text="contrast streching", height=2, highlightthickness=0,
                                     borderwidth=0,
                                     background=togglebtncolor, foreground='#E1D9D1',
                                     command=lambda: printlabel( funcs.contrast_stretching( image1 ) ) ).place( x=0,
                                                                                                                y=0 )
    threshold_btn = tk.Button( bit_toggle, width=20, text="Thresholding", height=2, highlightthickness=0,
                               borderwidth=0,
                               background=togglebtncolor, foreground='#E1D9D1',
                               command=lambda: thresholdpopup() ).place( x=0, y=37 )

    grayA1 = tk.Button( bit_toggle, width=20, text="Gray Level 1", height=2, highlightthickness=0, borderwidth=0,
                        background=togglebtncolor, foreground='#E1D9D1',
                        command=lambda: graylevelpopupA1() ).place( x=0, y=74 )
    grayA2 = tk.Button( bit_toggle, width=20, text="Gray level 2", height=2, highlightthickness=0, borderwidth=0,
                        background=togglebtncolor, foreground='#E1D9D1',
                        command=lambda: graylevelpopupA2() ).place( x=0, y=111 )
    bitplane = tk.Button( bit_toggle, width=20, text="bit plane", height=2, highlightthickness=0, borderwidth=0,
                          background=togglebtncolor, foreground='#E1D9D1',
                          command=lambda: bitplaneSlicing( cvim ) ).place( x=0, y=148 )


def onclickhisto():
    bgcolor = "#1e252b";
    histo_toggle = Frame( main_page, width=145, height=winheight, bg=bgcolor )
    histo_toggle.place( x=40, y=0 )
    histogram_equalization = tk.Button( histo_toggle, width=20, text="histogram equalization", height=2,
                                        highlightthickness=0, borderwidth=0,
                                        background=togglebtncolor, foreground='#E1D9D1',
                                        command=lambda: printlabel( funcs.histogram_equalization( cvim ) ) ).place( x=0,
                                                                                                                    y=0 )


def onclicklinear():
    bgcolor = "#1e252b";
    linear_toggle = Frame( main_page, width=145, height=1000, bg=bgcolor )
    linear_toggle.place( x=40, y=0 )
    log = tk.Button( linear_toggle, width=20, text="Log", height=2, highlightthickness=0,
                     borderwidth=0,
                     background=togglebtncolor, foreground='#E1D9D1',
                     command=lambda: printlabel( funcs.log( cvim ) ) ).place( x=0, y=0 )
    negative = tk.Button( linear_toggle, width=20, text="Negative", height=2, highlightthickness=0,
                          borderwidth=0,
                          background=togglebtncolor, foreground='#E1D9D1',
                          command=lambda: printlabel( funcs.Negative( cvim ) ) ).place( x=0, y=37 )

    inverseLog = tk.Button( linear_toggle, width=20, text="inverse log", height=2, highlightthickness=0, borderwidth=0,
                            background=togglebtncolor, foreground='#E1D9D1',
                            command=lambda: printlabel( funcs.inverseLog( cvim ) ) ).place( x=0, y=74 )
    Power_law_transformation = tk.Button( linear_toggle, width=20, text="Power law", height=2, highlightthickness=0,
                                          borderwidth=0,
                                          background=togglebtncolor, foreground='#E1D9D1',
                                          command=lambda: powerlawpopup() ).place( x=0, y=111 )
    identity = tk.Button( linear_toggle, width=20, text="identity", height=2, highlightthickness=0, borderwidth=0,
                          background=togglebtncolor, foreground='#E1D9D1',
                          command=lambda: printlabel( funcs.identity( cvim ) ) ).place( x=0, y=148 )


def onclickfilter():
    bgcolor = "#1e252b";
    filter_toggle = Frame( main_page, width=145, height=winheight, bg=bgcolor )
    filter_toggle.place( x=40, y=0 )
    median_filter = tk.Button( filter_toggle, width=20, text="median filter", height=2, highlightthickness=0,
                               borderwidth=0,
                               background=togglebtncolor, foreground='#E1D9D1',
                               command=lambda: printlabel( funcs.median_filter( image1 ) ) ).place( x=0, y=0 )
    composite = tk.Button( filter_toggle, width=20, text="composite laplacian", height=2, highlightthickness=0,
                           borderwidth=0,
                           background=togglebtncolor, foreground='#E1D9D1',
                           command=lambda: printlabel( funcs.compositeLaplacian( cvim ) ) ).place( x=0, y=37 )
    laplacian = tk.Button( filter_toggle, width=20, text="laplacian", height=2, highlightthickness=0,
                           borderwidth=0,
                           background=togglebtncolor, foreground='#E1D9D1',
                           command=lambda: printlabel( funcs.laplacian( cvim ) ) ).place( x=0, y=74 )
    sobel = tk.Button( filter_toggle, width=20, text="sobel", height=2, highlightthickness=0,
                       borderwidth=0,
                       background=togglebtncolor, foreground='#E1D9D1',
                       command=lambda: printlabel( funcs.sobel( cvim ) ) ).place( x=0, y=111 )


def onclickfreq():
    bgcolor = "#1e252b";
    freq_toggle = Frame( main_page, width=145, height=winheight, bg=bgcolor )
    freq_toggle.place( x=40, y=0 )
    bhfp = tk.Button( freq_toggle, width=20, text="BHPF", height=2, highlightthickness=0,
                      borderwidth=0,
                      background=togglebtncolor, foreground='#E1D9D1',
                      command=lambda: printcanvas( funcs.BHFP( cvim ) ) ).place( x=0, y=0 )

    blpf = tk.Button( freq_toggle, width=20, text="BLPF", height=2, highlightthickness=0,
                      borderwidth=0,
                      background=togglebtncolor, foreground='#E1D9D1',
                      command=lambda: printcanvas( funcs.BLPF( cvim ) ) ).place( x=0, y=37 )

    ihpf = tk.Button( freq_toggle, width=20, text="IHPF", height=2, highlightthickness=0,
                      borderwidth=0,
                      background=togglebtncolor, foreground='#E1D9D1',
                      command=lambda: printcanvas( funcs.IHPF( cvim ) ) ).place( x=0, y=76 )

    glpf = tk.Button( freq_toggle, width=20, text="GLPF", height=2, highlightthickness=0,
                      borderwidth=0,
                      background=togglebtncolor, foreground='#E1D9D1',
                      command=lambda: printcanvas( funcs.GLPF( imgpil ) ) ).place( x=0, y=113 )

    ilpf = tk.Button( freq_toggle, width=20, text="ILPF", height=2, highlightthickness=0,
                      borderwidth=0,
                      background=togglebtncolor, foreground='#E1D9D1',
                      command=lambda: printcanvas( funcs.ILPF( cvim ) ) ).place( x=0, y=150 )


def printlabel(image):
    im = Image.fromarray( (image).astype( np.uint8 ) )
    ph = ImageTk.PhotoImage( im )
    canvas2.create_image( 300, 210, image=ph )
    canvas2.image = ph


def histogramdisplay():
    f = Figure( figsize=(5, 5), dpi=100 )
    a = f.add_subplot( 111 )
    a.plot( histogram( image1 ) )
    canvas = FigureCanvasTkAgg( f, main_page )
    canvas.draw()
    canvas.get_tk_widget().pack( side=tk.BOTTOM, expand=True )


def histogram(img):
    img = image1
    rows = img.shape[0]
    columns = img.shape[1]
    count = np.zeros( 256 )
    for r in range( rows ):
        for c in range( columns ):
            gray_level = img[r, c]
            count[gray_level] += 1
    gray_levels = np.arange( 256 )
    f = Figure( figsize=(3, 2), dpi=80 )
    a = f.add_subplot( 111 )
    a.grid()
    a.set_ylabel( 'pixel count' )
    a.set_xlabel( 'gray levels' )
    a.plot( gray_levels, count )
    canvas = FigureCanvasTkAgg( f, main_page )
    canvas.draw()
    canvas.get_tk_widget().place( x=1110, y=50 )


def printcanvas(bw):
    f = Figure( figsize=(3, 3), dpi=100 )
    canvas = FigureCanvasTkAgg( f, main_page )
    a = f.add_subplot( 111 )
    a.imshow( bw, cmap='gray' )
    a.axis( 'off' )
    canvas.draw()
    canvas.get_tk_widget().place( x=180, y=400 )


def bitplaneSlicing(img):
    lst = []
    for i in range( img.shape[0] ):
        for j in range( img.shape[1] ):
            lst.append( np.binary_repr( img[i][j], width=8 ) )  # width = no. of bits

    eight_bit_img = (np.array( [int( i[0] ) for i in lst], dtype=np.uint8 ) * 128).reshape( img.shape[0], img.shape[1] )
    seven_bit_img = (np.array( [int( i[1] ) for i in lst], dtype=np.uint8 ) * 64).reshape( img.shape[0], img.shape[1] )
    six_bit_img = (np.array( [int( i[2] ) for i in lst], dtype=np.uint8 ) * 32).reshape( img.shape[0], img.shape[1] )
    five_bit_img = (np.array( [int( i[3] ) for i in lst], dtype=np.uint8 ) * 16).reshape( img.shape[0], img.shape[1] )
    four_bit_img = (np.array( [int( i[4] ) for i in lst], dtype=np.uint8 ) * 8).reshape( img.shape[0], img.shape[1] )
    three_bit_img = (np.array( [int( i[5] ) for i in lst], dtype=np.uint8 ) * 4).reshape( img.shape[0], img.shape[1] )
    two_bit_img = (np.array( [int( i[6] ) for i in lst], dtype=np.uint8 ) * 2).reshape( img.shape[0], img.shape[1] )
    one_bit_img = (np.array( [int( i[7] ) for i in lst], dtype=np.uint8 ) * 1).reshape( img.shape[0], img.shape[1] )
    f = Figure( figsize=(3, 3), dpi=100 )
    i1 = f.add_subplot( 241 )
    i1.imshow( one_bit_img, cmap='gray', aspect='auto' )
    i1.axis( 'off' )
    i2 = f.add_subplot( 242 )
    i2.imshow( two_bit_img, cmap='gray', aspect='auto' )
    i2.axis( 'off' )
    i3 = f.add_subplot( 243 )
    i3.imshow( three_bit_img, cmap='gray', aspect='auto' )
    i3.axis( 'off' )
    i4 = f.add_subplot( 244 )
    i4.imshow( four_bit_img, cmap='gray', aspect='auto' )
    i4.axis( 'off' )
    i5 = f.add_subplot( 245 )
    i5.imshow( five_bit_img, cmap='gray', aspect='auto' )
    i5.axis( 'off' )
    i6 = f.add_subplot( 246 )
    i6.imshow( six_bit_img, cmap='gray', aspect='auto' )
    i6.axis( 'off' )
    i7 = f.add_subplot( 247 )
    i7.imshow( seven_bit_img, cmap='gray', aspect='auto' )
    i7.axis( 'off' )
    i8 = f.add_subplot( 248 )
    i8.imshow( eight_bit_img, cmap='gray', aspect='auto' )
    i8.axis( 'off' )
    canvas = FigureCanvasTkAgg( f, main_page )
    canvas.draw()
    canvas.get_tk_widget().place( x=180, y=400 )



main_page.mainloop()  # Keep the window open
