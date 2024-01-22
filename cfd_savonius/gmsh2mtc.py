#!/usr/bin/env python
#
# GMSH2MTC
#
# Utilitaire de conversion des maillages gmsh (.msh) 2D et 3D en
# format mtc (.t).
# Fonctionne pour les maillages surfaciques et volumiques.
# Compatible uniquement avec des elements triangles/tetraedres.
#
# tommy.carozzani@mines-paristech.fr
# 12/2010

fichier_entree = "turbine.msh"
fichier_sortie = "turbine.t"


###########

print("Initialisation...")

f = open(fichier_entree)

position_flags = [ 0, 0, 0, 0, 0, 0 ]
flags = [ "$MeshFormat",
          "$EndMeshFormat",
          "$Nodes",
          "$EndNodes",
          "$Elements",
          "$EndElements" ]
ii = range(len(flags))

nb_noeuds = 0

connect_3d = list()
connect_2d = list()

###########

print("Recuperation position flags...")

t = f.readline()

while t:
    t = t.strip(" \t\n")
    for i in ii:
        if(t == flags[i]):
            position_flags[i] = f.tell()
            break
    t = f.readline()
    
print(position_flags)

###########

print("Traitement noeuds...")

f.seek(position_flags[2])

nb_noeuds = int(f.readline())

print("Nb de noeuds : "+str(nb_noeuds))

###########

print("Traitement connectivites...")

f.seek(position_flags[4])

t = f.readline()    # ligne ignoree (nb d'elements)
t = f.readline()

while (t and f.tell()!=position_flags[5]):
    t = t.split()
    if(len(t) <= 1):
        break

    if(t[1] == '2'):    # triangle
        i = 3 + int(t[2])
        lig = t[i] + " " + t[i+1] + " " + t[i+2] + " 0\n" 
        connect_2d.append(lig)

    if(t[1] == '4'):    # tetraedre
        i = 3 + int(t[2])
        lig = t[i] + " " + t[i+1] + " " + t[i+2] + " " + t[i+3] + "\n"
        connect_3d.append(lig)
        
    t = f.readline()

dim = 3
if(len(connect_3d) == 0):
    dim = 2

print("Nb elements 2d : "+str(len(connect_2d)))
print("Nb elements 3d : "+str(len(connect_3d)))
print("Dimension du maillage: "+str(dim))

###########

print("Ecriture .t ...")

fo = open(fichier_sortie,"w")

lig = str(nb_noeuds)+" "+str(dim)+" "+str(len(connect_2d)+len(connect_3d))+" "+str(dim+1)+"\n"
fo.write(lig)

# noeuds

f.seek(position_flags[2])
f.readline()    # ligne ignoree (nb noeuds)

i = 0
while i<nb_noeuds:
    t = f.readline()
    t = t.split(None)

    fo.write(t[1]+" "+t[2])
    if(dim==3):
        fo.write(" "+t[3])
    fo.write("\n")
    i += 1

# elements 3d

for e in connect_3d:
    fo.write(e)
    
# elements 2d

for e in connect_2d:
    if(dim==3):
        fo.write(e)
    else:
        fo.write(e.rsplit(None,1)[0] + "\n")

fo.close()


print("OK.")
