#!/usr/bin/env python3
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import style
from matplotlib import pyplot as plt

import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename, askopenfilename
from tkinter import ttk

import numpy as np

from diagram import Diagram
from wigner import CG
import csv

GUIVERSION = "0.5/2020.06.22"
Nmaxmax = 7

textshift = 0.01

muN = [0,0,0,0.35,0.45,0.45,0.45,0.40]
jvals=(0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5)
jvals_signed=(-5.5,-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5)


#theme = {"fg":"white", "bg":"black"}
#style.use("dark_background")
theme = {"fg":"black", "bg":"white"}
style.use("fast")
#style.use("ggplot")

def on_closing():
    plt.close("all")
    app.destroy()
    
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isbool(value):
  try:
    bool(value)
    return True
  except ValueError:
    return False


f = plt.figure()
f.set_tight_layout(True)

class NilssonApp(tk.Tk):
    global theme
    def __init__(self, *args, **kwargs):        
        """ App controller, defines frames and menues, stores the data """
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self,"Nilsson App")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.settings = {"kappa": tk.DoubleVar(self,0.05), "mu": tk.DoubleVar(self,0.35),
                         "Nd": tk.IntVar(self,50), "ranged": tk.DoubleVar(self,0.5),
                         "DeltaN2": tk.BooleanVar(self,True),
                         "verbose": tk.BooleanVar(self,False),
                         "gquench": tk.DoubleVar(self,0.7), "gR": tk.DoubleVar(self,0.35),
                         "reaction": tk.StringVar(self,"add"), "CG": tk.DoubleVar(self,0.0),
                         "Ii": tk.DoubleVar(self,0.0), "Ki": tk.DoubleVar(self,0.0),
                         "If": tk.DoubleVar(self,0.0), "Kf": tk.DoubleVar(self,0.0),
                         "j": tk.DoubleVar(self,0.0), "dK": tk.DoubleVar(self,0.0)
        }
        
        self.defaultvals = tk.BooleanVar(self, True)
        self.frame = MainWindow(container, self)
        self.frame.grid(row=0, column=0, sticky="nsew")
        self.frame.tkraise()
        self.frame.focus_set()

        def set_dark():
            
            global theme
            if theme["fg"] == "white":
                print("is dark")
                theme = {"fg":"black", "bg":"white"}
                style.use("ggplot")
            else:
                print("is light")
                theme = {"fg":"white", "bg":"black"}
                style.use("dark_background")
                
        menubar = tk.Menu(container)
        fileMenu = tk.Menu(menubar,tearoff=0)
        fileMenu.add_command(label="Save settings", command=self.save_set)
        fileMenu.add_command(label="Load settings", command=self.load_set)
        fileMenu.add_command(label="Save data as", command=self.frame.save_asdat)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command = quit)
        menubar.add_cascade(label="File", menu=fileMenu)
        
        optMenu = tk.Menu(menubar, tearoff=1)
        optMenu.add_checkbutton(label="Set verbose ouput", var=self.settings["verbose"])
        optMenu.add_command(label="Set \u03ba and \u03bc", command=self.set_kappamu)
        optMenu.add_command(label="Set range of \u03b4 values", command=self.set_delta)
        optMenu.add_checkbutton(label="Allow \u0394N=2 mixing", var=self.settings["DeltaN2"])
        optMenu.add_separator()
        optMenu.add_command(label="Options for g-factors", command=self.set_gfact)
        optMenu.add_command(label="Options for spec. factors", command=self.set_sfact)
        menubar.add_cascade(label="Options", menu=optMenu)

        displayMenu = tk.Menu(menubar,tearoff=0)
        displayMenu.add_checkbutton(label="Set dark theme", command=lambda: messagebox.showerror("Err", "Not implemented yet"))
        #displayMenu.add_checkbutton(label="Set dark theme", command=set_dark)
        menubar.add_cascade(label="Display", menu=displayMenu)
        
        helpMenu = tk.Menu(menubar, tearoff=0)
        helpMenu.add_command(label="Help", command=self.display_help)
        helpMenu.add_command(label="About", command=self.display_about)
        menubar.add_cascade(label="Help", menu=helpMenu)
        
        tk.Tk.config(self, menu=menubar)

    def save_set(self):
        """ Save the settings as a dat file """
        savefilepath = asksaveasfilename(filetypes=(("Data File", "*.dat"),("CSV file", "*.csv"),("Text File", "*.txt"),("All Files", "*.*")), 
            defaultextension='.dat', title="Save the settings")
        if not savefilepath:
            return
        with open(savefilepath, 'w') as outfile:  
            for key, value in self.settings.items():
                if value:
                    outfile.write("%s\t%s\n" % (key, value.get()))
                        
    def load_set(self):
        """ Load the settings from a file """
        openfilepath = askopenfilename(filetypes=(("Data File", "*.dat"),("CSV file", "*.csv"),("Text File", "*.txt"),("All Files", "*.*")), 
            defaultextension='.dat', title="Load the settings")
        if not openfilepath:
            return
        with open(openfilepath, 'r') as infile:  
            for l in infile:
                if l.split()[1].isdigit():
                    self.settings[l.split()[0]].set(int(l.split()[1]))
                elif isfloat(l.split()[1]):
                    self.settings[l.split()[0]].set(float(l.split()[1]))
                elif isbool(l.split()[1]):
                    self.settings[l.split()[0]].set(eval(l.split()[1]))
        self.frame.set_title()
        if self.settings["verbose"].get():
            print("settings loaded:")
            for key, value in self.settings.items():
                if value:
                    print(key, value.get())
            
    def display_help(self):
        helpwdw = tk.Toplevel(app)
        self.update()
        helpwdw.geometry("+%d+%d" % (self.winfo_rootx()+self.winfo_width()*0.5,self.winfo_rooty()+self.winfo_height()*0.5))
        helpwdw.wm_title("Help")
        label = tk.Label(helpwdw, text="For a basic calculation set N to your desired value and press \"Calculate\".")
        label.pack(side="top",fill="x",padx=10,pady=10)
        b = tk.Button(helpwdw, text="Exit",command=helpwdw.destroy)
        b.pack()
        helpwdw.mainloop()

    def display_about(self):
        aboutwdw = tk.Toplevel(app)
        self.update()
        aboutwdw.geometry("+%d+%d" % (self.winfo_rootx()+self.winfo_width()*0.5,self.winfo_rooty()+self.winfo_height()*0.5))
        aboutwdw.wm_title("About")
        label = tk.Label(aboutwdw, text="Nilsson model calculation GUI.\n Version %s \n K. Wimmer" % (GUIVERSION))
        label.pack(side="top",fill="x",padx=10,pady=10)
        b = tk.Button(aboutwdw, text="Exit",command=aboutwdw.destroy)
        b.pack()
        aboutwdw.mainloop()

    def set_kappamu(self):
        kappamuwdw = tk.Toplevel(app)
        self.update()
        kappamuwdw.geometry("+%d+%d" % (self.winfo_rootx()+self.winfo_width()*0.5,self.winfo_rooty()+self.winfo_height()*0.5))        
        kappamuwdw.wm_title("Set \u03ba and \u03bc")


        def toggle_defaults():
            if self.defaultvals.get():
                sca_kappa.config(state="disabled")
                txt_kappa.config(state="disabled")
                sca_mu.config(state="disabled")
                txt_mu.config(state="disabled")
                btn_reset.config(state="disabled")
                self.settings['kappa'].set(0.05)
                self.settings['mu'].set(muN[self.frame.Nmax.get()])
            else:
                sca_kappa.config(state="normal")
                txt_kappa.config(state="normal")
                sca_mu.config(state="normal")
                txt_mu.config(state="normal")
                btn_reset.config(state="normal")
            pass

        
        """ use default values for kappa and mu. """
        chb_default = tk.Checkbutton(kappamuwdw, text="use default values for \u03ba and \u03bc", var=self.defaultvals, command=toggle_defaults)
        chb_default.grid(row=0, column=0,padx=2.5,pady=2.5, sticky='W')
        
        
        
        fr_sel = tk.Frame(kappamuwdw)
        fr_sel.grid(row=1, column=0, sticky="ns")
        fr_sel.rowconfigure((0,1), weight=1)
        fr_sel.columnconfigure((0,1,2), weight=1)

        
        lbl_kappa = tk.Label(fr_sel, text="Set \u03ba (default 0.05)")
        sca_kappa = tk.Scale(fr_sel,from_=0.0, to=0.20, orient=tk.HORIZONTAL, digits = 3, resolution = 0.002,variable=self.settings['kappa'],state='disabled')
        sca_kappa.set(self.settings['kappa'].get())
        txt_kappa = tk.Entry(fr_sel, textvariable=self.settings['kappa'],state='disabled')
        
        lbl_kappa.grid(column = 0, row = 0, sticky = "s")
        sca_kappa.grid(column = 1, row = 0, sticky = "s")
        txt_kappa.grid(column = 2, row = 0, sticky = "s")
        
        lbl_mu = tk.Label(fr_sel, text="Set \u03bc (default 0.30)")
        sca_mu = tk.Scale(fr_sel,from_=0.0, to=1.0, orient=tk.HORIZONTAL, digits = 4, resolution = 0.005,variable=self.settings['mu'],state='disabled')
        sca_mu.set(self.settings['mu'].get())
        txt_mu = tk.Entry(fr_sel, textvariable=self.settings['mu'],state='disabled')
        
        lbl_mu.grid(column = 0, row = 1, sticky = "s")
        sca_mu.grid(column = 1, row = 1, sticky = "s")
        txt_mu.grid(column = 2, row = 1, sticky = "s")


        fr_but = tk.Frame(kappamuwdw)
        fr_but.grid(row=2, column=0, sticky="ns")
        fr_but.rowconfigure(0, weight=1)
        fr_but.columnconfigure((0,1,2), weight=1)
        
        def reset():
            variable=self.settings['kappa'].set(0.05)
            variable=self.settings['mu'].set(0.30)
            self.frame.set_title()
            
        btn_reset = tk.Button(fr_but,text="Reset",command=reset,state='disabled')
        btn_reset.grid(column=0,row=0,sticky="ew")
        btn_apply = tk.Button(fr_but,text="Apply",command=self.frame.set_title)
        btn_apply.grid(column=1,row=0,sticky="ew")
        btn_close = tk.Button(fr_but,text="Close",command=lambda:[self.frame.set_title(),kappamuwdw.destroy()])
        btn_close.grid(column=2,row=0,sticky="ew")
        kappamuwdw.mainloop()

    def set_delta(self):
        deltawdw = tk.Toplevel(app)
        self.update()
        deltawdw.geometry("+%d+%d" % (self.winfo_rootx()+self.winfo_width()*0.5,self.winfo_rooty()+self.winfo_height()*0.5))        
        deltawdw.wm_title("Set range of \u03b4 values")
        fr_sel = tk.Frame(deltawdw)
        fr_sel.grid(row=0, column=0, sticky="ns")
        fr_sel.rowconfigure((0,1), weight=1)
        fr_sel.columnconfigure((0,1,2), weight=1)
        
        lbl_Nd = tk.Label(fr_sel, text="Set number of \u03b4s")
        sca_Nd = tk.Scale(fr_sel,from_=1, to=100, orient=tk.HORIZONTAL, digits = 3, resolution = 1,variable=self.settings['Nd'])
        sca_Nd.set(self.settings['Nd'].get())
        txt_Nd = tk.Entry(fr_sel, textvariable=self.settings['Nd'])
        
        lbl_Nd.grid(column = 0, row = 0, sticky = "s")
        sca_Nd.grid(column = 1, row = 0, sticky = "s")
        txt_Nd.grid(column = 2, row = 0, sticky = "s")
        
        lbl_ranged = tk.Label(fr_sel, text="Set max \u03b4 (default 0.50)")
        sca_ranged = tk.Scale(fr_sel,from_=0.0, to=0.74, orient=tk.HORIZONTAL, digits = 2, resolution = 0.01,variable=self.settings['ranged'])
        sca_ranged.set(self.settings['ranged'].get())
        txt_ranged = tk.Entry(fr_sel, textvariable=self.settings['ranged'])
        
        lbl_ranged.grid(column = 0, row = 1, sticky = "s")
        sca_ranged.grid(column = 1, row = 1, sticky = "s")
        txt_ranged.grid(column = 2, row = 1, sticky = "s")

        fr_but = tk.Frame(deltawdw)
        fr_but.grid(row=1, column=0, sticky="ns")
        fr_but.rowconfigure(0, weight=1)
        fr_but.columnconfigure((0,1), weight=1)
        
        def reset():
            variable=self.settings['Nd'].set(50)
            variable=self.settings['ranged'].set(0.50)

        btn_reset = tk.Button(fr_but,text="Reset",command=reset)
        btn_reset.grid(column=0,row=0,sticky="ew")
        btn_close = tk.Button(fr_but,text="Close",command=deltawdw.destroy)
        btn_close.grid(column=1,row=0,sticky="ew")
        deltawdw.mainloop()

    def set_gfact(self):
        gfactwdw = tk.Toplevel(app)
        self.update()
        gfactwdw.geometry("+%d+%d" % (self.winfo_rootx()+self.winfo_width()*0.5,self.winfo_rooty()+self.winfo_height()*0.5))        
        gfactwdw.wm_title("Options for g-factors")
        fr_sel = tk.Frame(gfactwdw)
        fr_sel.grid(row=0, column=0, sticky="ns")
        fr_sel.rowconfigure((0,1), weight=1)
        fr_sel.columnconfigure((0,1,2), weight=1)
        
        lbl_quench = tk.Label(fr_sel, text="Quenching for g-factor")
        sca_quench = tk.Scale(fr_sel,from_=0, to=1, orient=tk.HORIZONTAL, digits = 3, resolution = 0.01,variable=self.settings['gquench'])
        sca_quench.set(self.settings['gquench'].get())
        txt_quench = tk.Entry(fr_sel, textvariable=self.settings['gquench'])
        
        lbl_quench.grid(column = 0, row = 0, sticky = "s")
        sca_quench.grid(column = 1, row = 0, sticky = "s")
        txt_quench.grid(column = 2, row = 0, sticky = "s")
        
        lbl_gR = tk.Label(fr_sel, text="Set value of g_R (default 0.35)")
        sca_gR = tk.Scale(fr_sel,from_=0.0, to=0.5, orient=tk.HORIZONTAL, digits = 2, resolution = 0.01,variable=self.settings['gR'])
        sca_gR.set(self.settings['gR'].get())
        txt_gR = tk.Entry(fr_sel, textvariable=self.settings['gR'])
        
        lbl_gR.grid(column = 0, row = 1, sticky = "s")
        sca_gR.grid(column = 1, row = 1, sticky = "s")
        txt_gR.grid(column = 2, row = 1, sticky = "s")

        fr_but = tk.Frame(gfactwdw)
        fr_but.grid(row=1, column=0, sticky="ns")
        fr_but.rowconfigure(0, weight=1)
        fr_but.columnconfigure((0,1), weight=1)
        
        def reset():
            variable=self.settings['gquench'].set(0.7)
            variable=self.settings['gR'].set(0.35)

        btn_reset = tk.Button(fr_but,text="Reset",command=reset)
        btn_reset.grid(column=0,row=0,sticky="ew")
        btn_close = tk.Button(fr_but,text="Close",command=gfactwdw.destroy)
        btn_close.grid(column=1,row=0,sticky="ew")
        gfactwdw.mainloop()

    def set_sfact(self):
        sfactwdw = tk.Toplevel(app)
        self.update()
        sfactwdw.geometry("+%d+%d" % (self.winfo_rootx()+self.winfo_width()*0.5,self.winfo_rooty()+self.winfo_height()*0.5))

        sfactwdw.wm_title("Options for spectroscopic factors")
        fr_sel = tk.Frame(sfactwdw)
        fr_sel.grid(row=0, column=0, sticky="ns")
        fr_sel.rowconfigure((0,1), weight=1)
        #fr_sel.columnconfigure((0,1,2), weight=1)
        lbl_rea = tk.Label(fr_sel, text="Select type of reaction")
        lbl_rea.grid(column = 0, row = 0, sticky = "ew",columnspan=6)
        
        """ Radiobuttons to select which type of reaction """
        rea_rem = ttk.Radiobutton(fr_sel, text = "nucleon removing: stripping, knockout", value="rem",var=self.settings["reaction"])
        rea_add = ttk.Radiobutton(fr_sel, text = "nucleon adding: pickup", value="add",var=self.settings["reaction"])
        rea_rem.grid(column = 0, row = 1, sticky = "ew",columnspan=6)
        rea_add.grid(column = 0, row = 2, sticky = "ew",columnspan=6)

        
        txt_CG = tk.Text(fr_sel, width=27, height=1, borderwidth=0,background=self.cget("background"))
        txt_CG.tag_configure("subscript", offset=-2)
        txt_CG.insert("insert", "<J", "", "i", "subscript", "K", "", "i", "subscript", "j\u0394K|J", "", "i", "subscript", "K", "", "i", "subscript" ,"> = ")
        txt_CG.grid(row=5,column=0,sticky="ns",columnspan=6)

        def update_CG():
            self.settings["CG"].set(CG(self.settings["Ii"].get(),self.settings["Ki"].get(),self.settings["j"].get(),self.settings["dK"].get(),self.settings["If"].get(),self.settings["Kf"].get()))
            txt_CG.delete(1.0, tk.END)
            txt_CG.insert("insert", "<J", "", "i", "subscript", "K", "", "i", "subscript", "j\u0394K|J", "", "f", "subscript", "K", "", "f", "subscript" ,"> = %.5f"% self.settings["CG"].get())
            
        
        """ Select spins """
        lbl_Ii = tk.Text(fr_sel, width=2, height=1, borderwidth=0,background=self.cget("background"))
        lbl_Ii.tag_configure("subscript", offset=-2)
        lbl_Ii.insert("insert", "I", "", "i", "subscript")
        spb_Ii = tk.Spinbox(fr_sel,values=jvals,width=5,textvariable=self.settings["Ii"],command=update_CG)
        lbl_Ii.grid(row=3,column=0,padx=2.5,pady=2.5,sticky="ns")
        spb_Ii.grid(row=3,column=1,padx=2.5,pady=2.5)
        
        lbl_If = tk.Text(fr_sel, width=2, height=1, borderwidth=0,background=self.cget("background"))
        lbl_If.tag_configure("subscript", offset=-2)
        lbl_If.insert("insert", "I", "", "f", "subscript")
        spb_If = tk.Spinbox(fr_sel,values=jvals,width=5,textvariable=self.settings["If"],command=update_CG)
        lbl_If.grid(row=3,column=2,padx=2.5,pady=2.5,sticky="ns")
        spb_If.grid(row=3,column=3,padx=2.5,pady=2.5)
        
        lbl_j = tk.Text(fr_sel, width=2, height=1, borderwidth=0,background=self.cget("background"))
        lbl_j.tag_configure("subscript", offset=-2)
        lbl_j.insert("insert", "j")
        spb_j = tk.Spinbox(fr_sel,values=jvals,width=5,textvariable=self.settings["j"],command=update_CG)
        lbl_j.grid(row=3,column=4,padx=2.5,pady=2.5,sticky="ns")
        spb_j.grid(row=3,column=5,padx=2.5,pady=2.5)
        
        lbl_Ki = tk.Text(fr_sel, width=2, height=1, borderwidth=0,background=self.cget("background"))
        lbl_Ki.tag_configure("subscript", offset=-2)
        lbl_Ki.insert("insert", "K", "", "i", "subscript")
        spb_Ki = tk.Spinbox(fr_sel,values=jvals,width=5,textvariable=self.settings["Ki"],command=update_CG)
        lbl_Ki.grid(row=4,column=0,padx=2.5,pady=2.5,sticky="ns")
        spb_Ki.grid(row=4,column=1,padx=2.5,pady=2.5)
        
        lbl_Kf = tk.Text(fr_sel, width=2, height=1, borderwidth=0,background=self.cget("background"))
        lbl_Kf.tag_configure("subscript", offset=-2)
        lbl_Kf.insert("insert", "K", "", "f", "subscript")
        spb_Kf = tk.Spinbox(fr_sel,values=jvals,width=5,textvariable=self.settings["Kf"],command=update_CG)
        lbl_Kf.grid(row=4,column=2,padx=2.5,pady=2.5,sticky="ns")
        spb_Kf.grid(row=4,column=3,padx=2.5,pady=2.5)
        
        lbl_dK = tk.Text(fr_sel, width=2, height=1, borderwidth=0,background=self.cget("background"))
        lbl_dK.tag_configure("subscript", offset=-2)
        lbl_dK.insert("insert", "\u0394K")
        spb_dK = tk.Spinbox(fr_sel,values=jvals_signed,width=5,textvariable=self.settings["dK"],command=update_CG)
        lbl_dK.grid(row=4,column=4,padx=2.5,pady=2.5,sticky="ns")
        spb_dK.grid(row=4,column=5,padx=2.5,pady=2.5)
        spb_dK.delete(0, tk.END)
        spb_dK.insert(0,0)
        
        fr_but = tk.Frame(sfactwdw)
        fr_but.grid(row=1, column=0, sticky="ns")
        fr_but.rowconfigure(0, weight=1)
        fr_but.columnconfigure((0,1), weight=1)
        
        def reset():
            variable=self.settings['Ii'].set(0.0)
            variable=self.settings['Ki'].set(0.0)
            variable=self.settings['If'].set(0.0)
            variable=self.settings['Kf'].set(0.0)
            variable=self.settings['j'].set(0.0)
            variable=self.settings['dK'].set(0.0)

        btn_reset = tk.Button(fr_but,text="Reset",command=reset)
        btn_reset.grid(column=0,row=0,sticky="ew")
        btn_close = tk.Button(fr_but,text="Close",command=sfactwdw.destroy)
        btn_close.grid(column=1,row=0,sticky="ew")
        sfactwdw.mainloop()
       
class MainWindow(tk.Frame):
    def __init__(self, parent, controller):
        """ Main window, plot Nilsson diagram """
        self.controller = controller
        tk.Frame.__init__(self,parent)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        self.Nmin = tk.IntVar(self,value=3)
        self.Nmax = tk.IntVar(self,value=3)

        self.title = tk.Label(self)
        self.set_title()
        self.title.grid(column = 0, row = 0, columnspan = 2, sticky = "ew")
        self.orbits = {}
        
        fr_menu = tk.Frame(self)
        fr_menu.grid(row=1, column=0, sticky="ns")
        fr_menu.columnconfigure((0,1), weight=1)


        """ Choose the oscillator shell for the calculation."""
        self.lbl_nmin = tk.Text(fr_menu, width=5, height=1, borderwidth=0,background=self.cget("background"))
        self.lbl_nmin.tag_configure("subscript", offset=-2)
        self.lbl_nmin.insert("insert", "N")
        self.lbl_nmin.grid(row=0,column=0,padx=2.5,pady=2.5,sticky="ns")
        self.spb_nmin = tk.Spinbox(fr_menu,from_=0,to=Nmaxmax,width=5,textvariable=self.Nmin,command=self.update_Nmin)
        self.spb_nmin.grid(row=0,column=1,padx=2.5,pady=2.5)

        
        
        """ Choose the maximum oscillator shell, disabled per default. """
        self.lbl_nmax = tk.Text(fr_menu, width=5, height=1, borderwidth=0,background=self.cget("background"))
        self.lbl_nmax.tag_configure("subscript", offset=-2)
        self.lbl_nmax.insert("insert", "N", "", "max", "subscript")
        self.lbl_nmax.grid(row=1,column=0,padx=2.5,pady=2.5,sticky="s")        
        self.spb_nmax = tk.Spinbox(fr_menu,state="disable",from_=self.Nmin.get(),to=Nmaxmax,width=5,textvariable=self.Nmax,command=self.update_Nmax)
        self.spb_nmax.grid(row=1,column=1,padx=2.5,pady=2.5)

        """ En/disable the range of N values. """
        self.endis_Nmax = tk.BooleanVar(self, False)
        self.chb_nrange = tk.Checkbutton(fr_menu, text="allow range of N", var=self.endis_Nmax, command=self.toggle_endis_Nmax)
        self.chb_nrange.grid(row=2, column=0,columnspan=2 ,padx=2.5,pady=2.5, sticky='W')

        

        """ Button to run the calculation. """
        btn_calc = tk.Button(fr_menu,text="Calculate",command=self.calculate)
        btn_calc.grid(row=3,column=1,columnspan=1,padx=2.5,pady=2.5,sticky="ew")

        w = ttk.Separator(fr_menu,orient=tk.HORIZONTAL).grid(row=4, columnspan=2,pady=2.5,sticky="ew")
        
        lbl_orb = ttk.Label(fr_menu,text="Select orbit")
        lbl_orb.grid(row=5,column=0,columnspan=2,padx=2.5,pady=2.5,sticky="ew")
        """ Select the orbit for plotting its properties """
        self.cmb_orb = ttk.Combobox(fr_menu, values=[""], state='disabled', width=10)
        self.cmb_orb.grid(row=6,column=0,columnspan=2,padx=2.5,pady=2.5,sticky="ew")
        self.cmb_orb.current(0)

        
        self.propopt = tk.StringVar(self,"wf")
        """ Radiobuttons to select which property to plot. """
        self.rad_wf = ttk.Radiobutton(fr_menu, text = "wave function", value="wf",var=self.propopt, state='disabled')
        self.rad_de = ttk.Radiobutton(fr_menu, text = "decoupling par", value="de",var=self.propopt, state='disabled')
        self.rad_gf = ttk.Radiobutton(fr_menu, text = "g-factor", value="gf",var=self.propopt, state='disabled')
        self.rad_sf = ttk.Radiobutton(fr_menu, text = "spec. factor", value="sf",var=self.propopt, state='disabled')
        self.rad_wf.grid(row=7,column=0,columnspan=2,padx=2.5,pady=2.5,sticky="ew")
        self.rad_de.grid(row=8,column=0,columnspan=2,padx=2.5,pady=2.5,sticky="ew")
        self.rad_gf.grid(row=9,column=0,columnspan=2,padx=2.5,pady=2.5,sticky="ew")
        self.rad_sf.grid(row=10,column=0,columnspan=2,padx=2.5,pady=2.5,sticky="ew")

        """ Button to plot the properties of the selcted orbit. """
        self.btn_plot = tk.Button(fr_menu,text="Plot",command=self.plot_prop, state='disabled')
        self.btn_plot.grid(row=20,column=1,columnspan=1,pady=2.5,padx=2.5,sticky="ew")
        
        
        """ Frame to display the calculation results. """
        fr_plot = tk.Frame(self)
        fr_plot.grid(row=1, column=1, sticky="nsew")
        fr_plot.columnconfigure(0, weight=1)
        fr_plot.rowconfigure(0, weight=1)
 
        self.axes = plt.subplot(111)
        self.canvas = FigureCanvasTkAgg(f, fr_plot)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0,column=0,sticky="nesw")
        self.canvas.callbacks.connect('pick_event', self.click_select)
        #f.set_tight_layout(True)
        plt.tight_layout()
        toolbarFrame = tk.Frame(master=fr_plot)
        toolbarFrame.grid(row=1,column=0,sticky="nesw")
        toolbar = CustomToolbar(self.canvas, toolbarFrame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand = False)
        fr_plot.rowconfigure(1, minsize = toolbarFrame.winfo_height())

    def set_title(self):
        if self.Nmax.get() == self.Nmin.get():
            ntext = "N = %d" % self.Nmax.get()
        else:
            ntext = "N = [%d,%d]" % (self.Nmin.get(),self.Nmax.get())
        self.title.config(text="Nilsson Diagram for %s, with \u03ba = %.3f, \u03bc = %.3f" % (ntext,self.controller.settings["kappa"].get(),self.controller.settings["mu"].get()))
        
        
    def update_Nmin(self):
        self.spb_nmax.config(from_=self.Nmin.get())
        if self.spb_nmax.cget('state') == "disabled":
            self.Nmax.set(self.Nmin.get())
        if self.controller.defaultvals.get():
            self.controller.settings["kappa"].set(0.05)
            self.controller.settings["mu"].set(muN[self.Nmax.get()])
        self.set_title()

    def update_Nmax(self):
        self.spb_nmin.config(to=self.Nmax.get())
        if self.controller.defaultvals.get():
            self.controller.settings["kappa"].set(0.05)
            self.controller.settings["mu"].set(muN[self.Nmax.get()])
        self.set_title()

    def toggle_endis_Nmax(self):
        """ Toggle to en- or dis-able the Nmax selector """
        if self.endis_Nmax.get():
            self.spb_nmax.config(state="normal")
            self.lbl_nmin.delete("1.0", tk.END)
            self.lbl_nmin.insert("insert", "N", "", "min", "subscript")
        else:
            self.spb_nmax.config(state="disabled")
            self.lbl_nmin.delete("1.0", tk.END)
            self.lbl_nmin.insert("insert", "N")
        self.update_Nmax()
        self.update_Nmin()
        
    def click_select(self, event):
        """ Define mouse clicks to select the orbit """
        if event.mouseevent.button == 1: 
            for l in range(self.diag.Nlev):           
                if event.artist==self.levels[l]: 
                    #self.selected = l
                    self.cmb_orb.current(l)
        if event.mouseevent.button == 3: 
            for l in range(self.diag.Nlev):           
                if event.artist==self.levels[l]: 
                    #self.selected = l
                    self.cmb_orb.current(l)
                    PlotProp(self,l,self.propopt.get())
                    
    def calculate(self):
        """ Calculate the nilsson diagram and wave functions """
        #print("calculating with kappa = %.3f, mu = %.3f" % (self.controller.settings["kappa"].get(),self.controller.settings["mu"].get()))
        self.set_title()
        self.levels = []
        self.diag = Diagram(kappa=self.controller.settings["kappa"].get(), mu=self.controller.settings["mu"].get(), Nd=self.controller.settings["Nd"].get(), ranged=self.controller.settings["ranged"].get(), Nmin = self.Nmin.get(), Nmax = self.Nmax.get(), DeltaN2=self.controller.settings["DeltaN2"].get(), verbose=self.controller.settings["verbose"].get())
        self.diag.rundiagram()
        for l in range(self.diag.Nlev):
            self.orbits[self.diag.plainNlabel(self.diag.nQN[l])] = l
        self.plot_diagram()
        self.update_orbselector()
        
    def plot_diagram(self):
        """ Plot the diagram """
        self.axes.clear()
        self.axes.plot([0,0],[np.min(self.diag.el),np.max(self.diag.el)],ls="--",linewidth=1,color=theme['fg'])
        for l in range(self.diag.Nlev):
            line, = self.axes.plot(self.diag.deltas,self.diag.el[l], ls="--" if self.diag.nQN[l][1] == 1 else "-",picker=5)
            self.axes.text(self.diag.deltas[-1]+textshift,self.diag.el[l][-1],"$%s$" % (self.diag.Nlabel(self.diag.nQN[l])), ha = 'left')
            self.axes.text(self.diag.deltas[0]-textshift,self.diag.el[l][0],"$%s$" % (self.diag.Nlabel(self.diag.nQN[l])), ha = 'right')
            self.levels.append(line)
        self.axes.set_xlabel('$\delta$',fontsize=12)
        self.axes.set_ylabel('$E/\hbar\omega$',fontsize=12)
        self.axes.tick_params(axis='both', which='major')
        self.axes.tick_params(axis='both', which='minor')
        self.axes.set_xlim(self.diag.deltas[0]-0.1, self.diag.deltas[-1]+0.1)

        self.canvas.draw()


    def update_orbselector(self):
        """ Update the orbit selector as set as active """
        options = [self.diag.plainNlabel(self.diag.nQN[l]) for l in range(self.diag.Nlev)]
        self.cmb_orb["values"] = options
        self.cmb_orb["state"] = "active"
        self.cmb_orb.current(0)
        self.btn_plot["state"] = "active"
        # radiobuttons
        self.rad_wf["state"] = "active"
        self.rad_de["state"] = "active"
        self.rad_gf["state"] = "active"
        self.rad_sf["state"] = "active"

    def plot_prop(self):
        if self.propopt.get() == "de" and self.diag.nQN[self.orbits[self.cmb_orb.get()]][0] != 1./2:
            messagebox.showerror("Err", "Decoupling parameter only for Omega = 1/2")
            return
        PlotProp(self, self.orbits[self.cmb_orb.get()],self.propopt.get())
        
    def save_asdat(self):
        """ Save the diagram as a txt file, with a header starting with # """
        savefilepath = asksaveasfilename(filetypes=(("Data File", "*.dat"),("Text File", "*.txt"),("All Files", "*.*")), 
            defaultextension='.dat', title="Save the data")
        if not savefilepath:
            return
        with open(savefilepath, "w") as output:
            if self.Nmax.get() == self.Nmin.get():
                ntext = "N = %d" % self.Nmax.get()
            else:
                ntext = "N = [%d,%d]" % (self.Nmin.get(),self.Nmax.get())
                output.write("#Nilsson Diagram for %s, with kappa = %.2f, mu = %.3f\n" % (ntext,self.controller.settings['kappa'].get(),self.controller.settings['mu'].get()) )
            output.write("#delta\t\t")
            for l in range(self.diag.Nlev):
                output.write("%s\t" % self.diag.plainNlabel(self.diag.nQN[l]))
            output.write("\n")
            np.savetxt(output, np.column_stack((self.diag.deltas,np.transpose(self.diag.el))),fmt='%.6f',delimiter='\t') 
       
class PlotProp():
    def __init__(self,parent, plotorb, prop):
        self.prop = prop
        self.diag = parent.diag
        self.plotorb = plotorb
        self.settings = parent.controller.settings
        
        self.plotwdw = tk.Toplevel(app)
        self.plotwdw.rowconfigure(0, weight=1)
        self.plotwdw.columnconfigure(0, weight=1)


        self.popupMenu = tk.Menu(self.plotwdw, tearoff=0)
        self.popupMenu.add_command(label="Save Figure",command=self.save_asfig)
        self.popupMenu.add_command(label="Save As .dat",command=self.save_asdat)
        self.popupMenu.add_command(label="Save As .xls",command=self.save_asxls)
        self.plotwdw.bind("<Button-3>", self.popup)
        
        
        self.fsub = plt.figure()
        self.fsub.set_tight_layout(True)
        self.canvas = FigureCanvasTkAgg(self.fsub, self.plotwdw)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0,column=0,sticky="nesw")
        
        toolbarFrame = tk.Frame(master=self.plotwdw)
        toolbarFrame.grid(row=1,column=0,sticky="nesw")
        toolbar = CustomToolbar(self.canvas, toolbarFrame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand = False)
        self.plotwdw.rowconfigure(1, minsize = toolbarFrame.winfo_height())

        if self.prop == "wf":
            self.plotwdw.wm_title("Wave function composition")
            self.plot_wf()

        if self.prop == "de":
            self.plotwdw.wm_title("Decoupling parameters")
            self.plot_de()

        if self.prop == "gf":
            self.plotwdw.wm_title("Gyromagnetic factor")
            self.plot_gf()

        if self.prop == "sf":
            self.plotwdw.wm_title("Spectroscopic factor")
            self.plot_sf()

    def popup(self, event):
        try:
            self.popupMenu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popupMenu.grab_release()

    def save_asfig(self):
        savefilepath = asksaveasfilename(filetypes=(("Portable Network Graphics", "*.png"),("Encapsulated Postscript", "*.eps"),("Portable Document Format", "*.pdf"),("All Files", "*.*")), 
            defaultextension='.png', title="Save the figure")
        if not savefilepath:
            return
        if savefilepath != "":
            plt.savefig(savefilepath)
            
    def save_asdat(self):
        savefilepath = asksaveasfilename(filetypes=(("Data File", "*.dat"),("Text File", "*.txt"),("All Files", "*.*")), 
            defaultextension='.dat', title="Save the data")
        if not savefilepath:
            return
        with open(savefilepath, "w") as output:
            if self.prop == "wf":
                output.write("#wave function composition for %s level\n" % self.diag.plainNlabel(self.diag.nQN[self.plotorb]))
                output.write("#delta\t\t")
                rwf, rqn = self.diag.wavefunction(self.plotorb)
                for l in range(len(rqn)):
                    output.write("%s\t\t" % self.diag.plainSlabel(rqn[l]))
                output.write("\n")
                np.savetxt(output, np.column_stack((self.diag.deltas,np.transpose(rwf))),fmt='%.6f',delimiter='\t')

            elif self.prop == "de":
                output.write("#decoupling parameters for %s level\n" % self.diag.plainNlabel(self.diag.nQN[self.plotorb]))
                a = self.diag.decoupling(self.plotorb)
                sz, gKN, gKP, gN, gP, bN, bP = self.diag.gfactors(self.plotorb,self.settings['gquench'].get(),self.settings['gR'].get())
                output.write("#delta\t\ta\t\tbN\t\tbP\n")
                np.savetxt(output, np.column_stack((self.diag.deltas,np.transpose(a),np.transpose(bN), np.transpose(bP))),fmt='%.6f',delimiter='\t') 
                
            elif self.prop == "gf":
                output.write("#g-factors for %s level\n" % self.diag.plainNlabel(self.diag.nQN[self.plotorb]))
                sz, gKN, gKP, gN, gP, bN, bP = self.diag.gfactors(self.plotorb,self.settings['gquench'].get(),self.settings['gR'].get())
                fN = bN*(gKN-self.settings['gR'].get())
                fP = bP*(gKP-self.settings['gR'].get())

                output.write("#delta\t\tsz\t\tgKN\t\tgKP\t\tgN\t\tgP\t\tbN\t\tbP\t\tbN(gK-gR)\tbP(gK-gR)\n")
                np.savetxt(output, np.column_stack(( self.diag.deltas, np.transpose(sz), np.transpose(gKN), np.transpose(gKP),
                                                     np.transpose(gN), np.transpose(gP),
                                                     np.transpose(bN), np.transpose(bP), np.transpose(fN), np.transpose(fP) ))
                           ,fmt='%.6f',delimiter='\t') 
            elif self.prop == "sf":
                output.write("#spectroscopic factors |c|^2 for %s level\n" % self.diag.plainNlabel(self.diag.nQN[self.plotorb]))
                output.write("#delta\t\t")
                rsf, rqn = self.diag.sfactors(self.plotorb)
                for l in range(len(rqn)):
                    output.write("%s\t\t" % self.diag.plainSlabel(rqn[l]))
                output.write("\n")
                np.savetxt(output, np.column_stack((self.diag.deltas,np.transpose(rsf))),fmt='%.6f',delimiter='\t')
                
    def save_asxls(self):
        pass
            
    def plot_wf(self):
        ax = plt.subplot(111)
        

        rwf, rqn = self.diag.wavefunction(self.plotorb)
        ax.set_title("wave function composition for $%s$ level" % self.diag.Nlabel(self.diag.nQN[self.plotorb]))
        ax.plot([0,0],[np.min(rwf),np.max(rwf)],ls="--",linewidth=1,color=theme['fg'])
        ax.plot([self.diag.deltas[0]-0.1,self.diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color=theme['fg'])
        for l in range(len(rqn)):
            ax.plot(self.diag.deltas,rwf[l])
            ax.text(self.diag.deltas[-1]+textshift,rwf[l][-1],"$%s$" % (self.diag.Slabel(rqn[l])), ha = 'left')
            ax.text(self.diag.deltas[0]-textshift,rwf[l][0],"$%s$" % (self.diag.Slabel(rqn[l])), ha = 'right')
        ax.set_xlim(self.diag.deltas[0]-0.1, self.diag.deltas[-1]+0.1)
        ax.tick_params(axis='both', which='major')
        ax.tick_params(axis='both', which='minor')
        ax.set_xlabel('$\delta$',fontsize=12)
        ax.set_ylabel('$c_{\Omega j}$',fontsize=12)
        self.canvas.draw()
        
    def plot_de(self):
        ax = []
        a = self.diag.decoupling(self.plotorb)
        sz, gKN, gKP, gN, gP, bN, bP = self.diag.gfactors(self.plotorb,self.settings['gquench'].get(),self.settings['gR'].get())
        if type(a) is not np.ndarray:
            return
        ax.append(plt.subplot(2,1,1))
        ax[-1].set_title("decoupling parameters for $%s$ level" % self.diag.Nlabel(self.diag.nQN[self.plotorb]))
        ax[-1].plot([0,0],[np.min(a),np.max(a)],ls="--",linewidth=1,color=theme['fg'])
        ax[-1].plot(self.diag.deltas,a)
        plt.setp(ax[-1].get_xticklabels(), visible=False)
        ax[-1].tick_params(axis='y', which='major')
        ax[-1].tick_params(axis='y', which='minor')
        ax[-1].set_ylabel('$a$',fontsize=12)

        ax.append(plt.subplot(2,1,2,sharex=ax[0]))
        mi = np.min([np.min(bN),np.min(bP)])
        ma = np.max([np.max(bN),np.max(bP)])

        ax[-1].plot([0,0],[mi,ma],ls="--",linewidth=1,color=theme['fg'])
        ax[-1].plot(self.diag.deltas,bN,label="neutron")
        ax[-1].plot(self.diag.deltas,bP,label="proton")
        ax[-1].tick_params(axis='both', which='major')
        ax[-1].tick_params(axis='both', which='minor')
        ax[-1].set_ylabel('$b_{0}$',fontsize=12)
        
        plt.legend(loc="best",fancybox=False,fontsize=12)
        
        ax[-1].set_xlim(self.diag.deltas[0]-0.1, self.diag.deltas[-1]+0.1)
        ax[-1].set_xlabel('$\delta$',fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(hspace=.0)
        self.fsub.align_ylabels(ax)
        
        self.canvas.draw()
        
    def plot_gf(self):
        ax = []
        if self.diag.nQN[self.plotorb][0] == 1./2:
            ax.append(plt.subplot2grid((4,1),(0,0)))
            ax.append(plt.subplot2grid((4,1),(1,0),sharex=ax[0]))
            ax.append(plt.subplot2grid((4,1),(2,0),sharex=ax[0]))
            ax.append(plt.subplot2grid((4,1),(3,0),sharex=ax[0]))
        else:
            ax.append(plt.subplot2grid((3,1),(0,0)))
            ax.append(plt.subplot2grid((3,1),(1,0),sharex=ax[0]))
            ax.append(plt.subplot2grid((3,1),(2,0),sharex=ax[0]))
            

        sz, gKN, gKP, gN, gP, bN, bP = self.diag.gfactors(self.plotorb,self.settings['gquench'].get(),self.settings['gR'].get())
        if type(sz) is not np.ndarray:
            return
        ax[0].set_title("$g$-factor for $%s$ level" % self.diag.Nlabel(self.diag.nQN[self.plotorb]))
        ax[0].plot([0,0],[np.min(sz),np.max(sz)],ls="--",linewidth=1,color=theme['fg'])
        ax[0].plot([self.diag.deltas[0]-0.1,self.diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color=theme['fg'])
        ax[0].plot(self.diag.deltas,sz)
        plt.setp(ax[0].get_xticklabels(), visible=False)
        ax[0].tick_params(axis='y', which='major')
        ax[0].tick_params(axis='y', which='minor')
        ax[0].set_ylabel('$<s_z>$',fontsize=12)
        
        #ax[1].plot([0,0],[np.min(jz),np.max(jz)],ls="--",linewidth=1,color=theme['fg'])
        #ax[1].plot([self.diag.deltas[0]-0.1,self.diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color=theme['fg'])
        #ax[1].plot(self.diag.deltas,jz)
        #plt.setp(ax[1].get_xticklabels(), visible=False)
        #ax[1].tick_params(axis='y', which='major')
        #ax[1].tick_params(axis='y', which='minor')
        #ax[1].set_ylabel('$<j_z>$',fontsize=12)
        
        mi = np.min([np.min(gKN),np.min(gKP)])
        ma = np.max([np.max(gKN),np.max(gKP)])
        ax[1].plot([0,0],[mi,ma],ls="--",linewidth=1,color=theme['fg'])
        ax[1].plot([self.diag.deltas[0]-0.1,self.diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color=theme['fg'])
        ax[1].plot(self.diag.deltas,gKN,label="neutron")
        ax[1].plot(self.diag.deltas,gKP,label="proton")
        plt.setp(ax[1].get_xticklabels(), visible=False)
        ax[1].tick_params(axis='y', which='major')
        ax[1].tick_params(axis='y', which='minor')
        ax[1].set_ylabel('$g_K$',fontsize=12)
        
        mi = np.min([np.min(gN),np.min(gP)])
        ma = np.max([np.max(gN),np.max(gP)])
        ax[2].plot([0,0],[mi,ma],ls="--",linewidth=1,color=theme['fg'])
        ax[2].plot([self.diag.deltas[0]-0.1,self.diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color=theme['fg'])
        ax[2].plot(self.diag.deltas,gN,label="neutron")
        ax[2].plot(self.diag.deltas,gP,label="proton")
        ax[2].set_xlim(self.diag.deltas[0]-0.1, self.diag.deltas[-1]+0.1)
        ax[2].tick_params(axis='y', which='major')
        ax[2].tick_params(axis='y', which='minor')
        ax[2].set_ylabel('$g$',fontsize=12)

        if self.diag.nQN[self.plotorb][0] == 1./2:
            plt.setp(ax[2].get_xticklabels(), visible=False)
            fN = bN*(gKN-self.settings['gR'].get())
            fP = bP*(gKP-self.settings['gR'].get())
            mi = np.min([np.min(fN),np.min(fP)])
            ma = np.max([np.max(fN),np.max(fP)])
            ax[3].plot([0,0],[mi,ma],ls="--",linewidth=1,color=theme['fg'])
            ax[3].plot([self.diag.deltas[0]-0.1,self.diag.deltas[-1]+0.1],[0,0],ls="--",linewidth=1,color=theme['fg'])
            ax[3].plot(self.diag.deltas,fN,label="neutron")
            ax[3].plot(self.diag.deltas,fP,label="proton")
            ax[3].set_ylabel('$b_0(g_K-g_R)$',fontsize=12)
            
        ax[-1].tick_params(axis='both', which='major')
        ax[-1].tick_params(axis='both', which='minor')
        ax[-1].set_xlabel('$\delta$',fontsize=12)
        plt.legend(loc="best",fancybox=False,fontsize=12)
        
        self.fsub.set_tight_layout(True)
        self.canvas.draw()
        self.fsub.subplots_adjust(hspace=0.0)
        
    def plot_sf(self):
        ax = []
        if self.settings["Ii"].get() == 0 and self.settings["Ki"].get() == 0 :
            ax.append(plt.subplot2grid((2,1),(0,0)))
            ax.append(plt.subplot2grid((2,1),(1,0),sharex=ax[0]))
        else:
            ax.append(plt.subplot2grid((3,1),(0,0)))
            ax.append(plt.subplot2grid((3,1),(1,0),sharex=ax[0]))
            ax.append(plt.subplot2grid((3,1),(2,0),sharex=ax[0]))

        rsf, rqn = self.diag.sfactors(self.plotorb)
        if self.settings["verbose"].get():
            print(self.settings["reaction"].get())

        g2 = 1
        if self.settings["Ii"].get() == 0 or self.settings["Kf"].get() == 0:
            g2 = 2

        ax[0].set_title("spectroscopic factors for $%s$ level" % self.diag.Nlabel(self.diag.nQN[self.plotorb]))
        ax[0].plot([0,0],[0.0,np.max(rsf)],ls="--",linewidth=1,color=theme['fg'])
        for l in range(len(rqn)):
            ax[0].plot(self.diag.deltas,rsf[l])
            ax[0].text(self.diag.deltas[-1]+textshift,rsf[l][-1],"$%s$" % (self.diag.Slabel(rqn[l])), ha = 'left')
            ax[0].text(self.diag.deltas[0]-textshift,rsf[l][0],"$%s$" % (self.diag.Slabel(rqn[l])), ha = 'right')
        plt.setp(ax[0].get_xticklabels(), visible=False)
        ax[0].tick_params(axis='y', which='major')
        ax[0].tick_params(axis='y', which='minor')
        ax[0].set_ylabel('$|c_{\Omega j}|^2$',fontsize=12)


        """ from Ii = 0 to any orbital """
        maxsf = 0
        for l in range(len(rqn)):
            jfact = 1
            if self.settings["reaction"].get() == "add":
                jfact = 1./(2*rqn[l][2]+1)
            dK = self.diag.nQN[self.plotorb][0]
            j = rqn[l][2]
            If = rqn[l][2]
            Kf = self.diag.nQN[self.plotorb][0]
            cg = CG(0,0,j,dK,If,Kf)
            if self.settings["verbose"].get():
                print("Ii",0,"Ki",0,"j", rqn[l][2], "dK",dK, "If",If, "Kf", Kf)
                print("CG",cg,"g2",g2,"jfact",jfact)

            sf = jfact*g2*cg**2*rsf[l]
            maxsf = max(np.max(sf),maxsf)
            ax[1].plot(self.diag.deltas,sf)
            ax[1].text(self.diag.deltas[-1]+textshift,sf[-1],"$%s$" % (self.diag.Slabel(rqn[l])), ha = 'left')
            ax[1].text(self.diag.deltas[0]-textshift,sf[0],"$%s$" % (self.diag.Slabel(rqn[l])), ha = 'right')
        ax[1].plot([0,0],[0.0,maxsf],ls="--",linewidth=1,color=theme['fg'],zorder=-10)
        ax[1].tick_params(axis='y', which='major')
        ax[1].tick_params(axis='y', which='minor')
        ax[1].set_ylabel(r'$S(0^+\rightarrow j)$',fontsize=12)
        if self.settings["verbose"].get():
            print("----------------------")
        if self.settings["Ii"].get() != 0 or self.settings["Ki"].get() != 0 :
            plt.setp(ax[1].get_xticklabels(), visible=False)
            Ii = self.settings["Ii"].get()
            Ki = self.settings["Ki"].get()
            If = self.settings["If"].get()
            Kf = self.settings["Kf"].get()
            jfact = 1
            if self.settings["reaction"].get() == "add":
                jfact = (2*Ii+1)/(2*If+1)
            """ from Ii to any orbital """
            maxsf = 0
            for l in range(len(rqn)):
                j = rqn[l][2]
                dK = self.settings["Kf"].get() - self.settings["Ki"].get()
                cg = CG(Ii,Ki,j,dK,If,Kf)
                if self.settings["verbose"].get():
                    print("Ii",Ii,"Ki",Ki,"j", rqn[l][2], "dK",dK, "If",If, "Kf", Kf)
                    print("CG",cg,"g2",g2,"jfact",jfact)
                #if cg == 0: ## causes colors to be wrong
                #    continue
                sf = jfact*g2*cg**2*rsf[l]
                maxsf = max(np.max(sf),maxsf)
                ax[2].plot(self.diag.deltas,sf,linestyle="-" if self.settings["j"].get()==0 or j == self.settings["j"].get() else "--")
                ax[2].text(self.diag.deltas[-1]+textshift,sf[-1],"$%s$" % (self.diag.Slabel(rqn[l])), ha = 'left')
                ax[2].text(self.diag.deltas[0]-textshift,sf[0],"$%s$" % (self.diag.Slabel(rqn[l])), ha = 'right')
            ax[2].plot([0,0],[0.0,maxsf],ls="--",linewidth=1,color=theme['fg'],zorder=-10)
            ax[2].tick_params(axis='y', which='major')
            ax[2].tick_params(axis='y', which='minor')
            if Ii.is_integer() and not If.is_integer():
                ax[2].set_ylabel(r'$S(%d \rightarrow %d/2)$' %(int(Ii),int(2*If)),fontsize=12)
            if not Ii.is_integer() and If.is_integer():
                ax[2].set_ylabel(r'$S(%d/2 \rightarrow %d)$' %(int(2*Ii),int(If)),fontsize=12)
                
        #print(jfact)

        #mi = np.min([np.min(bN),np.min(bP)])
        #ma = np.max([np.max(bN),np.max(bP)])
        #
        #ax[-1].plot([0,0],[mi,ma],ls="--",linewidth=1,color=theme['fg'])
        #ax[-1].plot(self.diag.deltas,bN,label="neutron")
        #ax[-1].plot(self.diag.deltas,bP,label="proton")
        #ax[-1].tick_params(axis='both', which='major')
        #ax[-1].tick_params(axis='both', which='minor')
        #ax[-1].set_ylabel('$b_{0}$',fontsize=12)
        #
        #plt.legend(loc="best",fancybox=False,fontsize=12)
        
        ax[-1].set_xlim(self.diag.deltas[0]-0.1, self.diag.deltas[-1]+0.1)
        ax[-1].set_xlabel('$\delta$',fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(hspace=.0)
        self.fsub.align_ylabels(ax)
        
        self.canvas.draw()
        
            
class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self,canvas_,parent_):
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            #('Back', 'Back to  previous view', 'back', 'back'),
            #('Forward', 'Forward to next view', 'forward', 'forward'),
            #(None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            #(None, None, None, None),
            #('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        NavigationToolbar2Tk.__init__(self,canvas_,parent_)

        
app = NilssonApp()
app.protocol("WM_DELETE_WINDOW", on_closing)
app.mainloop()
