def plot_parallel_hists(Counts_Array, Bins_Array, Labels, Colors, xlabel, title, filename, Lines=None, xlim=None):
    import numpy as np
    import matplotlib.pyplot as plt
    if not len(Colors)==len(Labels)==np.shape(Counts_Array)[0]:
        print(len(Colors),len(Labels),np.shape(Counts_Array)[0])
        print("Dimensions do not correspond.\n Size of Colors and Labels must correspond to the number of histograms to be plotted, i.e. the number of lines in Counts_Array.")
        return None
    if not np.shape(Counts_Array)[1]==np.shape(Counts_Array)[1]:
        print(np.shape(Counts_Array)[1],np.shape(Counts_Array)[1])
        print("Dimensions do not correspond.\n Abscissa (nb of nc in Bins_Array) and ordonnates (nb of nc in Counts_Array) do not correspond).")
        return None
    
    plt.figure(figsize=(20,10))
    ax = plt.subplot(1,1,1)
    plt.xlabel(xlabel,fontsize=15) #A M
    plt.yticks([],fontsize=20)
    plt.xticks(fontsize=30)
    plt.title(title,fontsize=25,pad=30)
    plt.yticks([])
    
    J=[0.65,0.65,0.65,0.65,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7]
    jump=0
    nl,nc=np.shape(Counts_Array)#nb de hists, nb de bins par hists
    gap=np.max(Counts_Array) +1/nl*np.max(Counts_Array)
    
    print("len J = ",len(J))
    print("len labels = ", len(Labels))
    
    for l in range(0,nl):
        hist=np.zeros((nc,2))#On crée un hist virtuel ici, devrait être notre histogram
        hist[:,0]=Bins_Array[l][1:]
        hist[:,1]=Counts_Array[l]+jump
        ax.plot([np.min(Bins_Array),np.max(Bins_Array)],[jump,jump],color='black')
        ax.plot(hist[:,0],hist[:,1], linewidth = 2, linestyle = 'solid',color=Colors[l])
        ax.fill_between(hist[:,0],jump,hist[:,1], alpha=0.3,color=Colors[l])
        plt.text(np.min(Bins_Array)+0.1, jump+J[l]*gap, Labels[l], fontsize = 13) #TO MODIFY IF NECESSARY
        jump-=gap
    
    if Lines!=None:
        Linescolors=['r','g','orange','b','purple','c']
        for li in range(len(Lines)):
            plt.axvline(Lines[li][1],label=Lines[li][0],linestyle='--',color=Linescolors[li])
    if xlim != None:
        plt.xlim(xlim)
    plt.legend(bbox_to_anchor=(0.01, l/(2*nl)),fontsize=20)   
    plt.grid(linestyle='--',linewidth=0.8)
    plt.savefig(filename) 
    plt.show()



##################################



def polardensityplot(Y, R, title='', bins=100, cmap='twilight',save=False, file='polarplot.png'):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Attention: Il y a des effets de bord au niveau de 0° - 360° ; Ils seront particulièrement visibles si bins est petit.
    #a = np.linspace(0, 2*np.pi, bins); b = np.linspace(0, 60, bins)
    #c = np.histogram2d(R, Y, bins=bins, density=True)[0]
    
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111, polar=True)
    ax.grid(True, linewidth=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    #ctf = ax.contourf(a, b, c, cmap=cmap)
    #plt.colorbar(ctf)
    ax.bar(x = np.radians(18), height=60,width=np.radians(36),color="blue", alpha=0.25) # Red area
    ax.bar(x = np.radians(162),height=60,width=np.radians(36),color="red",alpha=0.25) # Blue area
    
    ax.scatter(np.radians(Y),[r/1 for r in R])
    
    angle_text = [18,54,90,126,162,198,234,270,306,342]
    y_text = 63 #Nucleic acid conformation
    conformation = ["C3' endo","C4' exo", "O4' endo","C1' exo", "C2' endo",
                  "C3' exo", "C4' endo","O4' exo", "C1' endo","C2' exo" ]
    
    for i in range(0, len(angle_text)):
        if angle_text[i] < 180:
            ha_text="left"
        else:
            ha_text="right"
        ax.text(x=np.radians(angle_text[i]),y=y_text,s=conformation[i],fontweight="bold",ha=ha_text,fontsize=15)
        
    #ticks radius Vmax
    ax.set_rmax(60);ax.set_rticks([0,20,40,60], fontsize=15)
    ax.set_rlabel_position(108)
    ax.tick_params(axis="y",which="both",labelsize=15)
    
    #ticks circle angle P
    ax.set_xticks(np.radians(np.linspace(0,360,10,endpoint=False)),minor=False)
    plt.setp(ax.xaxis.get_majorticklabels()[1:5],ha="left")
    plt.setp(ax.xaxis.get_majorticklabels()[6:9],ha="right")
    
    #remove duplicate
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    plt.legend(by_label.values(), by_label.keys(),loc="upper right", markerscale=2, fontsize=10, bbox_to_anchor=(1.2, 1))
    plt.title(title,fontsize=24)
    plt.suptitle(t="Pseudorotation Phase angles P(polar) and Amplitude Vmax (radius)")
    plt.tight_layout()
    plt.savefig(file) if save else None
    plt.show()