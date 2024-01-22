# Generic imports
import os
import sys
import math
import time
import numpy as np
import gmsh

################################
### Environment for Wind Turbine 
class savonius():

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'savonius'
        self.act_size = 4
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        # variables: camber and 3 thickness
        self.x_min    = np.array([0.1,0.02,0.02,0.02])
        self.x_max    = np.array([0.65,0.08,0.08,0.08])
        self.x_0      = 0.5*(self.x_min+self.x_max)
        self.path     = path
        self.cp_moy= 0
        self.cp_max= 0

        self.n_points = 8
        self.M = 0.3
        self.T = 0.12
        self.P = 0.4
        self.Xi = -1 #leading edge point x
        self.Xf = 0.0 #trailing edge point x
        self.angle = 0 #inclinaison
        self.x_corde = np.arange(self.Xi, self.Xf+0.025, 0.025)
        self.y_corde = np.zeros(len(self.x_corde))

        # Set episode number
        self.episode  = 0

    ### CFD resolution
    def cfd_solve(self, x, ep):

        # Create folders and copy cfd (please kill me)
        # On met les résultats là dedans 
        self.output_path = self.path+'/'+str(ep)+'/'  # Pour chaque épisode
        self.vtu_path    = self.output_path+'vtu/'
        self.effort   = self.output_path+'effort/'
        self.msh_path   = self.output_path+'msh/'
        self.t_mesh_path   = self.output_path+'t_mesh/'
        
        os.makedirs(self.effort)
        os.makedirs(self.vtu_path)
        os.makedirs(self.msh_path)
        os.makedirs(self.t_mesh_path)
        
        os.system('cp -r cfd_savonius ' + self.output_path + '.')   # A quoi sert cette ligne ???
        
        # Convert action to coordinates 		
        control_parameters = np.array(x)
        
        # Generation of NACA profile
        self.M = control_parameters[0]
        y_camber = self.y_c(self.x_corde)
        longueur_part_camber = np.zeros(0)
        cumule_longueur_part_camber = np.zeros(0)
        for i in range(len(self.x_corde)-1):
            longueur_part_camber = np.append(longueur_part_camber,((y_camber[i+1] - y_camber[i])**2+(self.x_corde[i+1] - self.x_corde[i])**2)**0.5)
            cumule_longueur_part_camber = np.append(cumule_longueur_part_camber,np.sum(longueur_part_camber[:-1]) + longueur_part_camber[i])

        longueur_camber = np.sum(longueur_part_camber)
        part_longueur = longueur_camber/(self.n_points/2)

        new_points = np.zeros((int(self.n_points/2 + 1),2))
        new_points[0,0] = self.x_corde[0]
        new_points[-1,0] = self.x_corde[-1]
        for i in range((int(self.n_points/2 + 1)-2)):
            dist_ref = (i+1)*part_longueur
            ref = abs(cumule_longueur_part_camber[0] - dist_ref)
            for j in range(len(cumule_longueur_part_camber)-1):
                new_points[i+1,0] = self.x_corde[j+1]
                new_points[i+1,1] = y_camber[j+1]
                dist = cumule_longueur_part_camber[j+1]
                if (abs(dist - dist_ref) < ref):
                    new_points[i+1,0] = self.x_corde[j+1]
                    new_points[i+1,1] = y_camber[j+1]
                    ref = abs(dist - dist_ref)
                else:
                    break
        new_x_corde = new_points[:,0]
        new_y_corde = np.zeros(len(new_x_corde))
        new_y_camber = self.y_c(new_x_corde)
        gradient = self.dy_dx(new_x_corde)
        thickness = self.y_t(new_x_corde)
        courbure = self.teta(gradient)

        for i in range(3):
            thickness[i+1] = control_parameters[i+1]
        x_upper = self.x_u(new_x_corde,new_y_camber, thickness,courbure)
        x_lower = self.x_l(new_x_corde,new_y_camber, thickness,courbure)
        y_upper = self.y_u(new_x_corde,new_y_camber, thickness,courbure)
        y_lower = self.y_l(new_x_corde,new_y_camber, thickness,courbure)
        n_control_pts = int(2*len(new_x_corde)-2)
        horz_control_pts = np.zeros((n_control_pts,2))
        for i in range(len(new_x_corde)):
            if (i==0):
                horz_control_pts[i,0] = new_x_corde[i]
                horz_control_pts[i,1] = new_y_corde[i]
            elif (i==(len(new_x_corde)-1)):
                horz_control_pts[i,0] = new_x_corde[i]
                horz_control_pts[i,1] = new_y_corde[i]
            else:
                horz_control_pts[i,0] = x_upper[i]
                horz_control_pts[i,1] = y_upper[i]
                horz_control_pts[-i,0] = x_lower[i]
                horz_control_pts[-i,1] = y_lower[i]

        control_pts = np.zeros((n_control_pts,2))
        unordered = np.zeros((n_control_pts,2))
        for i in range(len(horz_control_pts)):
            unordered[i,0] = horz_control_pts[i,0]*math.cos(math.radians(self.angle)) - horz_control_pts[i,1]*math.sin(math.radians(self.angle))
            unordered[i,1] = horz_control_pts[i,0]*math.sin(math.radians(self.angle)) + horz_control_pts[i,1]*math.cos(math.radians(self.angle))

        control_pts[0] = unordered[int(n_control_pts/2)]
        control_pts[int(n_control_pts/2)] = unordered[0]
        for i in range(int(n_control_pts/2 - 1)):
            control_pts[i+1] = unordered[int(n_control_pts/2 - i - 1)]
            control_pts[i+(int(n_control_pts/2)+1)] = unordered[-i-1]

        
        mesh_size = 0.005 # Mesh size
        try:
            # Init GMSH
            gmsh.initialize(sys.argv)
            # Ask GMSH to display information in the terminal
            gmsh.option.setNumber("General.Terminal", 1)

            # Create a model and name it "shape"
            model = gmsh.model
            model.add("shape")        
            
            shapepoints = []
            for j in range(len(control_pts)):
                shapepoints.append(model.geo.addPoint(control_pts[j,0], control_pts[j,1], 0.0,mesh_size))
            shapepoints.append(shapepoints[0])

            

            # Curveloop using splines
            shapespline = model.geo.addSpline(shapepoints)
            model.geo.addCurveLoop([shapespline],1)
            
            

            # Surface  
            model.geo.addPlaneSurface([1],1)
            


            # This command is mandatory and synchronize CAD with GMSH Model. The less you launch it, the better it is for performance purpose
            model.geo.synchronize()

            # gmsh version 2.0
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)

            # Mesh (2D)
            model.mesh.generate(2)
            # Write on disk
            gmsh.write(self.output_path+'cfd_savonius/airfoil.msh')

            # Finalize GMSH
            gmsh.finalize()

        except Exception as e:
            # Finalize GMSH
            gmsh.finalize()
            print('error: ', e)
            pass

        # convert to .t
        os.system('cd '+self.output_path+'cfd_savonius ; python3 gmsh2mtc.py')
        os.system('cd '+self.output_path+'cfd_savonius ; cp -r airfoil.msh ../msh')
        os.system('cd '+self.output_path+'cfd_savonius ; module load cimlibxx/master')
        os.system('cd '+self.output_path+'cfd_savonius ; echo 0 | mtcexe airfoil.t')
        os.system('cd '+self.output_path+'cfd_savonius ; cp -r airfoil.t ../t_mesh')
        
        # Solve problem using cimlib and move vtu and drag folder
        os.system('cd '+self.output_path+'cfd_savonius/.; touch run.lock; mpirun -n 8 /softs/cemef/cimlibxx/master/bin/cimlib_CFD_driver Principale.mtc > trash.txt;')
        os.system('mv '+self.output_path+'cfd_savonius/Resultats/2d/* '+self.vtu_path+'.')
        os.system('mv '+self.output_path+'cfd_savonius/Resultats/Efforts.txt '+self.effort+'.')
        os.system('rm -r '+self.output_path+'cfd_savonius')
        
        # Save
        os.system('cp -r '+self.vtu_path+'bulles_00450.vtu ./video/')
        os.system('mv ./video/bulles_00450.vtu '+'./video/video_'+str(self.episode)+'.vtu')
  
        # Compute reward
        with open('./cfd_savonius/Resultats/Efforts.txt', 'r') as f:
            next(f) # Skip header
            for line in f:
                L_finesse = [] 
                f.readline()
                for ligne in f :
                    a,b, cx, cy = ligne.split()
                    cx, cy = -float(cx), -float(cy)
                    L_finesse.append(cy/cx)
            finesse = np.array(L_finesse)  # 
        

        begin_take_reward = 150 # When we begin to take the reward 


        # Compute new reward
        self.reward = finesse[begin_take_reward:].mean()
        self.finesse_moy = finesse[begin_take_reward:].mean()
        self.finesse_max = finesse[begin_take_reward:].max()

        
        # On écrit dans le gros fichier
        # Write (d, L, R, e, Reward, Cp) in a txt
        print(os.path)
        if not os.path.isfile('Values.txt'):
            f = open('Values.txt','w')
            f.write('Index'+'\t'+'Camber'+'\t'+'T1'+'\t'+'T2'+'\t'+'T3'+'\t'+'finesse_moy'+'\t'+'finesse_max'+'\t'+'Reward'+'\n')
        else:
            f = open('Values.txt','a')
        f.write(str(self.episode)+'\t'+str(self.M)+'\t'+str(control_parameters[1])+'\t'+str(control_parameters[2])+'\t'+str(control_parameters[3])+'\t'+str(self.finesse_moy)+'\t'+str(self.finesse_max)+'\t'+str(self.reward)+'\n')
        f.close()
		
        self.episode      += 1 #new
        
        return self.reward

    ### Take one step
    def step(self, actions, ep):

        conv_actions = self.convert_actions(actions)
        reward       = self.cfd_solve(conv_actions, ep)

        return reward, conv_actions

    ### Provide observation
    def observe(self):

        # Always return the same observation
        return self.obs

    ### Convert actions
    def convert_actions(self, actions):

        # Convert actions
        conv_actions  = self.act_size*[None]
        x_p           = self.x_max - self.x_0
        x_m           = self.x_0   - self.x_min

        for i in range(self.act_size):
            if (actions[i] >= 0.0):
                conv_actions[i] = self.x_0[i] + x_p[i]*actions[i]
            if (actions[i] <  0.0):
                conv_actions[i] = self.x_0[i] + x_m[i]*actions[i]

        return conv_actions

    ### Close environment
    def close(self):
        pass

    ### A function to replace text in files
    ### This function finds line containing string, erases the
    ### whole line it and replaces it with target
    def line_replace(self, string, line, target):

        command = "sed -i '/"+string+"/c\\"+line+"' "+target
        os.system(command)
    
    
    ### NACA Generation functions
    def y_c(self, x):
        y_camber = np.zeros(0)
        for i in range(len(x)):
            if (x[i]<(self.Xi + self.P)):
                y_camber = np.append(y_camber,(self.M/(self.P)**2)*(2*self.P*(x[i]-self.Xi) - (x[i]-self.Xi)**2))
            else:
                y_camber = np.append(y_camber,(self.M/(1 - self.P)**2)*(1 - 2*self.P + 2*self.P*(x[i]-self.Xi) - (x[i]-self.Xi)**2))            
        return y_camber

    def dy_dx(self, x):
        derive = np.zeros(0)
        for i in range(len(x)):
            if (x[i]<(self.Xi + self.P)):
                derive = np.append(derive,(self.M/(self.P)**2)*(2*self.P - 2*(x[i]-self.Xi)))
            else:
                derive = np.append(derive,(self.M/(1 - self.P)**2)*(2*self.P - 2*(x[i]-self.Xi)))            
        return derive

    def y_t(self, x):
        mi_thickness = np.zeros(0)
        for i in range(len(x)):
            mi_thickness = np.append(mi_thickness,(self.T/0.2)*(0.2969*(x[i]-self.Xi)**0.5 - 0.126*(x[i]-self.Xi) - 0.3516*(x[i]-self.Xi)**2 + 0.2843*(x[i]-self.Xi)**3 - 0.1036*(x[i]-self.Xi)**4))#0.1015
        return mi_thickness

    def teta(self, x):
        angle = np.zeros(0)
        for i in range(len(x)):
            angle = np.append(angle,math.atan(math.radians(x[i])))
        return angle

    def x_u(self, x,y,thick,t):
        x_cood_u = np.zeros(0)
        for i in range(len(x)):
            x_cood_u = np.append(x_cood_u,x[i] - thick[i]*math.sin(math.radians(t[i])))
        return x_cood_u

    def y_u(self, x,y,thick,t):
        y_cood_u = np.zeros(0)
        for i in range(len(x)):
            y_cood_u = np.append(y_cood_u,y[i] + thick[i]*math.cos(math.radians(t[i])))
        return y_cood_u

    def x_l(self, x,y,thick,t):
        x_cood_l = np.zeros(0)
        for i in range(len(x)):
            x_cood_l = np.append(x_cood_l,x[i] + thick[i]*math.sin(math.radians(t[i])))
        return x_cood_l

    def y_l(self, x,y,thick,t):
        y_cood_l = np.zeros(0)
        for i in range(len(x)):
            y_cood_l = np.append(y_cood_l,y[i] - thick[i]*math.cos(math.radians(t[i])))
        return y_cood_l
