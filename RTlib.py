import numpy as np                        # need everywhere
import matplotlib.pyplot as plt           # need everywhere
import matplotlib.cm as cm                # color gradation
import scipy.special as special           # for modeling klopfenstein
import scipy.integrate as integrate       # for moleling klopfenstein
from scipy.special import gamma           # for modeling klopfenstein
from tqdm.notebook import tqdm            # progress bar (for jupyter, if you use local .py you may need to remove ".notebook")
import pandas as pd
import datetime
import os


def Laser_history(end):
    """
    Plot the history of our laser machine
    How to use?
    ======================
    end = 2026.5    
    Laser_history(end)
    ======================
    """
    def Makearr_anotate(ll,lw,al,aw,direction):
        x = np.array([0,ll-al,ll-al,ll,ll-al,ll-al,0,0])
        y = np.array([-lw,-lw,-lw-aw,0,lw+aw,lw,lw,-lw])/2

        if direction == 'horizontal':
            return x,y
        elif direction == 'vertical':
            return y,x

    laser = ['Minimaster','Pharos','CarbideJPN-1','CarbideJPN-2','CarbideUMN']
    power = np.array([3,15,40,40,80])
    start = np.array([2017 + 3/12,2018 + 4/12,2019 + 10/12,2023 + 10/12,2024])
    length = np.array([end - 2017 + 3/12, 
                       2019 + 10/12-2018 + 4/12, 
                       2023 + 10/12-2019 - 10/12, 
                       end - 2023 + 10/12,
                       end - 2024])
    color = ['m','gold','orange','orangered','red']
    label = ['power','start','period','color']
    d_today = datetime.date.today()
    today = float(d_today.strftime('%Y'))+float(d_today.strftime('%m'))/12+float(d_today.strftime('%d'))/364

    hist_df = pd.DataFrame([power,start,length,color],index = label, columns = laser)

    lw = np.ones(5)*2
    al = np.ones(5)*0.5
    aw = np.ones(5)*1.5
    direction = 'horizontal' # or vertical


    arrx,arry = np.zeros(5),np.zeros(5)
    fig = plt.figure(figsize = (9,6))
    ax = fig.add_subplot(111)
    ax.axvline(today,color = 'k',ls = 'dashed',lw = 0.5,zorder = 0)
    for i in range(0,len(laser)):
        arrx,arry = Makearr_anotate(length[i],lw[i],al[i],aw[i],direction)
        ax.plot(arrx+start[i],arry+power[i],color = color[i])
        ax.text((arrx+start[i])[-2],(arry+power[i])[-2],laser[i],va = 'bottom',fontsize = 15,zorder = 1)
        ax.fill(arrx+start[i],arry+power[i],color = color[i],alpha = 0.9,zorder = 1)


    ax.set_axisbelow(True)
    ax.grid()
    ax.set_ylim(0,100)
    ax.set_xlim(2017,2026.2)
    ax.set_xlabel('Year',fontsize = 15)
    ax.set_ylabel('Averaged power [W]',fontsize = 15)
    ax.tick_params(labelsize = 13)
    fig.tight_layout()
    plt.savefig('Laser_history.png',dpi = 400)    

class Bandwidth_class:
    def __init__(self):
        pass
        
    def LiteBIRD_LFT(self):
        vc = np.array([40,50,60,68,78,89,100,119,140])*1e+9
        return vc
    
    def Taurus(self):
        return np.array([220,400])*1e+9

class Transmission_lib:
    '''
    * Def name
        __init__
    * Description
        set global values
    * input parameters

    * Defined global values
        - mu:  vacuum permeability [m kg s-2 A-2]
        - ep0: vacuum permitivity [m-3 kg-1 s4 A2]
        - c:   speed of light [m/s]
        - pi:  pi    
    '''
    def __init__(self):        
        # ==========================
        # Constants
        # - - - - - - - - - - - - - - -
        self.mu = 1.25663706e-06  # vacuum permeability [m kg s-2 A-2]
        self.ep0 = 8.8542e-12     # vacuum permitivity [m-3 kg-1 s4 A2]
        self.c = 2.9979e+08       # speed of light [m/s]
        self.pi = np.pi           # pi
        # - - - - - - - - - - - - - - -
        # ==========================
        
    def Diffraction_boundary_pitch(self,n,v,theta,bound):
        '''
        * Def name
            Diffraction_boundary_pitch
        * Description
            Show pitch given material and upper frequency
        * input parameters
            - n: refractive index of material
            - v: upper frequency of the bandwidth [Hz]
            - theta: maximum incident angle of the telescope [deg]
            - bound: SWS shape, rectangular or hexagon
        * return
            - pitch [mm]
        '''
        theta_rad = np.radians(theta)
        if bound == 'rectangular':
            p = self.c/v/(n+np.sin(theta_rad))*1e+3
            print('Pitch [mm] = %.3f'%p)
        elif bound == 'hexagon':
            p = self.c/v/(n+np.sin(theta_rad))*2/np.sqrt(3)*1e+3
            print('Pitch [mm] = %.3f'%p)
        else:
            print('bound should be rectangular or hexagon')
            p = None
        return p

    def Klopfenstein(self,h,num,ni,ns,Gamma):
        '''
        * Def name
            Klopfenstein
        * Description
            Calculate Klofenstein index profile
            Based on two papers:
                1: Klopfenstein(1956):https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4051841
                2: Grann et al(1995): https://opg.optica.org/view_article.cfm?pdfKey=fbf947c3-8dd3-440b-bfe51c58d6e98f89_33114
        * input parameters
            - h:   height of the taper along z axis [arbitral unit]
            - num: number of layer (as same as resolution of depth of SWS)
            - ni:  refractive index of air, so 1.0
            - ns:  refractive index of substrate
            - Gamma: important parameter to control the trade-off between the ripple in the opera- tion bands and bandwidth

        * return
            - n_arr: Klopfenstein index profile
            - z_arr: array of depth (height)
            - z_space: array of thickness each layer
        '''
        z_arr = np.linspace(0,h,num)            # define array of depth (height)
        n_arr = np.zeros(num)                   # set array of Klopfenstein index profile
        z_space = np.ones(len(z_arr))*(h/num)   # calculate array of thickness each layer
        rho_0 = 0.5*np.log(ns/ni)               # calcualte rho_0(see Klopfenstein(1956))
        A = np.arccosh(rho_0/Gamma)             # calculate A(see Klopfenstein(1956))
        
        # Calculate Klopfenstein index profile 
        # (see Eq.12 in Klopfenstein(1956), and Eq.6 in Grann et al(1995))
        for i in range(0,num):
            x_i = 2.*z_arr[i]/h-1.
            phi_int = integrate.quad(lambda y: special.i1(A*np.sqrt(1.-y**2))/(A*np.sqrt(1.-y**2)), 0, x_i)  
            n_arr[i] = np.sqrt(ni*ns) * np.exp(Gamma * A**2 * phi_int[0])
        return n_arr,z_arr,z_space
        
    def Brauer_emt_anti_symmetric(self,freq,n1,n2,f1,f2,p1,p2):
        '''
        * Def name
            Brauer_emt_anti_symmetric
        * Description
            Calculate effective refractive index given area fraction
            Based on Brauer(1994): https://opg.optica.org/view_article.cfm?pdfKey=92654043-e0c1-4520-98b312f00908c33d_42237
        * input parameters
            - freq:   inpu frequency [Hz]
            - n1: refractive index of air
            - n2: refractive index of substrate, extraordinary
            - f1: area fraction along x axis
            - f2: area fraction along y axis
            - p1: pitch x
            - p2: pitch y

        * return
            - n_: 0th ordered effective refractive index
            - neff: 2nd ordered effective refractive index
        '''
        
        lamda = self.c/freq   # wavelength
        f = (f1+f2)/2.        # average fraction in x and y

        e1 = n1**2.*self.ep0  # refractive index --> permitivity (air) 
        e2 = n2**2.*self.ep0  # refractive index --> permitivity (substrate)

        ell_0 = (1.0 - f1)*e1+f1*e2    # Eq.1
        els_0 = 1./((1.-f2)/e1+f2/e2)  # Eq.2

        ell_2 = ell_0*(1.+(np.pi**2/3.)*(p1/lamda)**2.*f1**2*(1.-f1)**2.*((e2-e1)**2./(self.ep0*ell_0)))                    # Eq.3
        els_2 = els_0*(1.0+(np.pi)**2/3.0*(p2/lamda)**2*f2**2*(1.-f2)**2.*((e2-e1)**2.)*ell_0/self.ep0*(els_0/(e2*e1))**2.) # Eq.4
        
        e_2nd_up = (1.0 - f1)*e1 + f1*els_2           # Eq.6
        e_2nd_down = 1./((1.0 - f2)/e1 + f2/ell_2)    # Eq.7

        n_=(1-f**2)*n1+f**2*n2                        # Eq.5
        n__2nd_up = np.sqrt(e_2nd_up/(self.ep0))      # permitivity --> refractive index (up)
        n__2nd_down = np.sqrt(e_2nd_down/(self.ep0))  # permitivity --> refractive index (down)

        neff = 0.2*(n_+2.0*n__2nd_up+2.0*n__2nd_down) # Eq.8
        return n_*np.ones(len(neff)),neff
    
    def fit_oblique_basic_multilayer_r_t_incloss(self, n, losstan, d, freq_in, angle_i, incpol):
        '''
        * Def name
            fit_oblique_basic_multilayer_r_t_incloss
        * Description
            Calculate coefficient of reflectance and transmittance based on Transfer matrix method, made by Tomo Matsumura
            (modified with this class by RTakaku)
        * input parameters
            - n:  refractive index of substrate
            - losstan: loss tangent of substrate
            - d: thickness [m] 
            - freq_in: input frequency [Hz]
            - angle_i: incident angle [rad]
            - incpol: 1 for s-state, E field perpendicular to the plane of incidnet, -1 for P-state, E in the plane of incident

        * return
            - output ([0]: freq, [1]: coeff of reflectance, [2]: coeff of transmittance (complex numpy array))
        '''
        num=len(d) #; the number of layer not including two ends
        const = np.sqrt((self.ep0)/(4.*self.pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

        # ;-----------------------------------------------------------------------------------
        # ; angle of refraction
        angle = np.zeros(num+2)          # ; angle[0]=incident angle
        angle[0] = angle_i
        for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

        # ;-----------------------------------------------------------------------------------
        # ; define the frequency span
        l = len(freq_in)
        output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)

        # ;-----------------------------------------------------------------------------------
        # ; define the effective thickness of each layer
        h = np.zeros(num,'complex')
        n_comparr = np.zeros(len(n),'complex')
        n_comparr[0] = complex(n[0], -0.5*n[0]*losstan[0])
        n_comparr[num+1] = complex(n[num+1], -0.5*n[num+1]*losstan[num+1])

        # ;-----------------------------------------------------------------------------------
        # ; for loop for various thickness of air gap between each layer
        for j in range(0,l):
            for i in range(0,num): 
                n_comparr[i+1] = complex(n[i+1], -0.5*n[i+1]*losstan[i+1])
                h[i] = n_comparr[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer

            freq = freq_in[j]
            k = 2.*self.pi*freq/self.c

            # ;===========================================
            # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
            Y = np.zeros(num+2,'complex')
            for i in range(0,num+2):
                if (incpol == 1):
                    Y[i] = const*n_comparr[i]*np.cos(angle[i])
                    cc = 1.
                if (incpol == -1):
                    Y[i] = const*n_comparr[i]/np.cos(angle[i])
                    cc = np.cos(angle[num+1])/np.cos(angle[0])

            # ;===========================================
            # ; define matrix for single layer
            m = np.identity((2),'complex')    # ; net matrix
            me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
            for i in range(0,num):
                me[0,0] = complex(np.cos(k*h[i]), 0.)
                me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
                me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
                me[1,1] = complex(np.cos(k*h[i]), 0.)
                m = np.dot(m,me)

            r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
            t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

            output[0,j] = freq+0.j #; unit of [Hz]
            output[1,j] = r
            output[2,j] = t

        return output
    
    
    def Optimization_height_gamma(self,freq,harr,Gammao,d,n0,input_n,input_losstan,num,angle_i,incpol):
        '''
        * Def name
            Optimization_height_gamma
        * Description
            Calculate transmission for each height and gamma given frequency
        * input parameters
            - freq: frequency [Hz], 1D numpy array
            - harr: height [mm], 1D numpy array
            - Gammao: gamma in Klopfenstein, 1D numpy array
            - d: thickness [mm], float
            - n0: refractive index environment, float
            - input_n: refractive index material, float
            - input_losstan: loss tangent material, float
            - num: number of layer of Klopfenstein index profile, integer
            - angle_i: incident angle of light [deg], float
            - incpol: 1 for s-state, E field perpendicular to the plane of incidnet, -1 for P-state, E in the plane of incident
        * return
            - trans_arr[number of height, number of gamma, number of frequency], 3D numpy array
        '''
        
        trans_arr = np.empty([len(harr),len(Gammao),len(freq)])
        angle_rad = np.radians(angle_i)

        for hi in range(0,len(harr)):
            count = 0
            h = harr[hi]*1e-3
            input_d = d*1e-3 - h*2
            print('input thickness = %.3f mm'%(input_d*1e+3),'{0}/{1}'.format(hi+1,len(harr)))
            for i in tqdm(range(0,len(Gammao)),desc = 'depth = %.2f mm'%(h*1e+3)+', Table...{0}/{1}'.format(hi+1,len(harr))):
                no_arr, z, d_klop = self.Klopfenstein(h,num,n0,input_n,Gammao[i]) # n, z, thickness each layer
                n_klop = np.concatenate((np.array([n0]),no_arr))

                thickness = np.concatenate((d_klop,np.array([input_d]),d_klop))
                index = np.concatenate((n_klop,np.array([input_n]),n_klop[::-1]))
                losstan = np.concatenate((np.ones(len(n_klop))*input_losstan,np.array([input_losstan]),np.ones(len(n_klop))*input_losstan))

                trans_i = self.fit_oblique_basic_multilayer_r_t_incloss(index, losstan, thickness, freq, angle_rad, incpol)
                if count == 0:
                    trans_arri = np.abs(trans_i[2])**2
                    n_arr = no_arr
                    count += 1
                else:
                    trans_arri = np.vstack((trans_arri,np.abs(trans_i[2])**2))
                    n_arr = np.vstack((n_arr,no_arr))
                    count +=1
            trans_arr[hi] = trans_arri
        return trans_arr
    
        
    def Design_multi_layer(self,n0,n,vc):
        '''
        * Def name
            Design_multi_layer
        * Description
            Calculate refractive index and thickness for multi-layer ARC (signal, two and three layer)
        * input parameters
            - n0: refractive index environment, float
            - n: refractive index of material, float
            - vc: center frequency [Hz], float
        * return
            - n_single: refractive index, single layer ARC 1D numpy array
            - n_two   : refractive index, two layer ARC 1D numpy array
            - n_three : refractive index, three layer ARC 1D numpy array
            - d_single: thickness[m], single layer ARC 1D numpy array
            - d_two   : thickness[m], two layer ARC 1D numpy array
            - d_three : thickness[m], three layer ARC 1D numpy array
        '''
        # Single_ARC
        n1_single = np.sqrt(n0*n)
        
        # Two-layer ARC
        n1_two = (n0**2 * n)**(0.25)
        n2_two = (n0 * n**3)**(0.25)
        
        #Three-layer ARC
        n1_three = (n0**3 * n)**(0.25)
        n2_three = (n0**2 * n**2)**(0.25)
        n3_three = (n0**1 * n**3)**(0.25)
        
        n_single = np.array([n1_single])
        n_two = np.array([n1_two,n2_two])
        n_three = np.array([n1_three,n2_three,n3_three])
        
        # thickness
        lamda = self.c/vc
        d_single = lamda/4/n_single
        d_two = lamda/4/n_two
        d_three = lamda/4/n_three
        
        return n_single, n_two, n_three, d_single, d_two, d_three
    
    
    def Calculate_transmittance_multilayer_bothsides(self,freq,n_arr,d_arr,n0,ns,ds,input_lostan,angle_i,incpol):
        '''
        * Def name
            Calculate_transmittance_multilayer_bothsides
        * Description
            Calculate transmittance/reflectance, for multi-layer ARC (basically signal, two and three layer, but can be applied for multi-layer more)
        * input parameters
            - freq: input frequency [Hz], 1D numpy array
            - narr: array of multi-layer part, 1D numpy array
            - d_arr: thickness of multi-layer part, 1D numpy array
            - n0: refractive index of environment, float
            - ns: refractive index of material, float
            - ds: thickness of material, float
            - input_lostan: loss tangent of material, float
            - angle_i: incident angle [deg], float
            - incpol: 1 for s-state, E field perpendicular to the plane of incidnet, -1 for P-state, E in the plane of incident
        * return
            - freq[Hz], reflection coeff, transmission coeff
        '''        
        index = np.concatenate((np.array([n0]),n_arr,np.array([ns]),n_arr[::-1],np.array([n0])))
        thickness = np.concatenate((d_arr,np.array([ds]),d_arr[::-1]))
        losstan = np.concatenate((np.ones(len(n_arr)+1)*input_lostan, np.array([input_lostan]),np.ones(len(n_arr)+1)*input_lostan))
        
        freq_comp, refl, trans = self.fit_oblique_basic_multilayer_r_t_incloss(index, losstan, thickness, freq, angle_i, incpol)
        return freq_comp, refl, trans
        
    def Find_maximum_gamma(self,vc,n0,input_n,P,f_ang_thre,n_select,d_select,n_offset):
        '''
        * Def name
            Find_maximum_gamma
        * Description
            Find maximum gamma for given index and thickness.
        * input parameters
            - vc: center frequency of given frequency band [Hz], float
            - n0: refractive index of environment, float
            - input_n: refractive index of material, float
            - P: Pitch of structures [mm], float
            - f_ang_thre: threshold of the angle [deg], float
            - n_select: refractive index, 1D numpy array
            - d_select: thickness, 1D numpy array
            - n_offset: offset of the refractive index, 1D array
        * return
            - gamma_array: gamma array, 1D numpy array
        '''
        f = np.linspace(0.0,1.0,10001)
        n_2OEMT = self.Brauer_emt_anti_symmetric(vc,n0,input_n,f,f,P*1e-3,P*1e-3)[1]
        Gamma = np.arange(0.01,0.71,0.01)
        gamma_arr = np.empty(len(n_select))

        for i in range(0,len(n_select)):
            count = 0
            f_ang = 0
            while f_ang <= f_ang_thre:

                n_arr,d_arr,z_arr = self.Klopfenstein(d_select[i],num,n0,input_n,Gamma[count])

                w1 = np.sqrt(f[np.abs(n_arr[0]+n_offset[i] - n_2OEMT).argmin()]*P**2)
                w2 = np.sqrt(f[np.abs(n_arr[-1]+n_offset[i] - n_2OEMT).argmin()]*P**2)

                f_ang = np.degrees(np.arctan((d_select[i]*1e+3)/(w2-w1)))

                count += 1
            count_f = count - 1
            gamma_arr[i] = Gamma[count_f]
        return gamma_arr  
    
    def Find_width(self,vc,n0,input_n,P,n_arr,z_arr):
        f = np.linspace(0.0,1.0,10001)
        n_2OEMT = self.Brauer_emt_anti_symmetric(vc,n0,input_n,f,f,P*1e-3,P*1e-3)[1]
        wi_arr = np.empty(len(n_arr))
        for i in range(0,len(n_arr)):
            wi_arr[i] = np.sqrt(f[np.abs(n_arr[i]-n_2OEMT).argmin()]*P**2)
            
        print('Top width= [mm]',wi_arr[0])
        top_width = wi_arr[0]
        
        wp2 = wi_arr/2
        w_arr = np.concatenate((-wp2[::-1],wp2))
        w_arr = np.concatenate((w_arr-P,w_arr,w_arr+P))
        
        d_arr = np.concatenate((z_arr[::-1],z_arr))
        d_arr = np.concatenate((d_arr,d_arr,d_arr))
        return w_arr,d_arr,top_width
    
    def Plot_find_gamma(self,trans_thre,Gammao,t_ave,harr,dir_name,save_name):

        Optimal_height = np.zeros(len(trans_thre))
        Optimal_gamma = np.zeros(len(trans_thre))
        Optimal_ind = np.zeros([len(trans_thre),2])

        fig = plt.figure(figsize = (8,5))
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\Gamma_m$',fontsize = 18)
        ax.set_ylabel(r'$T_{ave}$',fontsize = 18)
        ax.tick_params(labelsize = 13)
        ax.grid(True)
        ax.set_ylim(0.95,1.001)

        color = ['r','b','g','y','c','m']
        for i in range(0,len(trans_thre)):
            try:
                ind = np.where(t_ave>=trans_thre[i])
                t_ave_thre = t_ave[ind[0][0]]
                ax.plot(Gammao,t_ave_thre,color[i],label = 'Depth = %.2f mm'%harr[ind[0][0]]+', $T_{ave}>$%s'%trans_thre[i])
                ax.axvline(Gammao[np.argmax(t_ave_thre)],ls = '--',lw = 1,color = color[i])

                Optimal_height[i] = harr[ind[0][0]]
                Optimal_gamma[i] = Gammao[np.argmax(t_ave_thre)]
                Optimal_ind[i] = np.array([ind[0][0],np.argmax(t_ave_thre)])
            except Exception as e:
                print('skip',e)
        ax.legend()
        fig.tight_layout()
        plt.savefig(dir_name+'/'+save_name+'.png')

        return Optimal_height,Optimal_gamma,  Optimal_ind

    
    def Save_trans_single_Klop(self,freq,gamma_arr,h_arr,w_arr,d_arr,opt_ind, input_n, input_d, trans, dir_name,save_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        np.savez(dir_name+'/'+save_name+'.npz',
                 freq = freq, 
                 gamma_arr = gamma_arr, 
                 h_arr = h_arr, 
                 w_arr = w_arr, 
                 d_arr = d_arr,
                 opt_ind = opt_ind,
                 input_n = input_n, 
                 input_d = input_d, 
                 trans = trans)