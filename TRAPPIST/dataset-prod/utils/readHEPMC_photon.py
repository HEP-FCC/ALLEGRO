import random
import math
import argparse

parser = argparse.ArgumentParser(description="Read HepMC3 file")

parser.add_argument(
    "--input-file",
    type=str,
    required=True,
    help="Path to HepMC3 file"
)

parser.add_argument(
    "--output-file",
    type=str,
    required=True,
    help="Path to output HepMC3 file"
)

parser.add_argument(
    "--evt_per_file",
    type=int,
    help="Number of events per file"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)

# Parse arguments
args = parser.parse_args()
inputfile = args.input_file
outputfile = args.output_file
evt_per_file = args.evt_per_file
random.seed(args.seed)

# Open file
f = open(inputfile, "r")

##print("File:", inputfile)
##print("Events per file:", evt_per_file)
##print("Seed:", args.seed)

list_PnV=[]
# pi0
list_pi0_px=[]
list_pi0_py=[]
list_pi0_pz=[]
list_pi0_E=[] # change this to a list of float
list_pi0_mass=[]
# pi0 vertex
list_vx=[]
list_vy=[]
list_vz=[]
list_vt=[]
# tau
list_tau_px=[]
list_tau_py=[]
list_tau_pz=[]
list_tau_E=[]
list_tau_mass=[]
list_tau_id=[]

for line in f:
    line_list=line.split(" ")
    if line_list[0]=="E":
        # analyse the previous event
        for i_line in range(len(list_PnV)):
            # pi0: is a particle && PDGID=111
            if list_PnV[i_line][0]=="P" and list_PnV[i_line][3]=="111": 
                
                # find the pi0 vertex
                for j_line in range(len(list_PnV)): 
                    if list_PnV[j_line][0]=="V" and list_PnV[j_line][1]==list_PnV[i_line][2]:
                        
                        # find tau
                        for k_line in range(len(list_PnV)):
                            #if list_PnV[k_line][0]=="P" and list_PnV[k_line][1]==list_PnV[j_line][3][1:-1]:
                            if list_PnV[k_line][0]=="P" and list_PnV[k_line][1]==list_PnV[j_line][3][1:-1] and list_PnV[k_line][3]==random.choice(["15", "-15"]):
                                # keep only the single pi0 decay mode
                                count_pi0=0
                                for l_line in range(len(list_PnV)):
                                    if list_PnV[l_line][0]=="P" and list_PnV[l_line][3]=="111" and list_PnV[l_line][2]==list_PnV[j_line][1]:
                                        count_pi0=count_pi0+1
                                        #print(list_PnV[l_line][1], count_pi0)
                                if count_pi0!=1:
                                    continue
                                # calculate pi0 eta
                                this_px=float(list_PnV[i_line][4])
                                this_py=float(list_PnV[i_line][5])
                                this_pz=float(list_PnV[i_line][6])
                                this_E=math.sqrt(this_px*this_px + this_py*this_py + this_pz*this_pz)
                                if math.acos(this_pz/this_E)>0.65 and math.acos(this_pz/this_E)<2.4915927 :
                                    # fill pi0
                                    list_pi0_px.append(list_PnV[i_line][4])
                                    list_pi0_py.append(list_PnV[i_line][5])
                                    list_pi0_pz.append(list_PnV[i_line][6])
                                    list_pi0_E.append(this_E)
                                    list_pi0_mass.append("0.0000000000000000e+00")
                                    # fill vertex
                                    list_vx.append(list_PnV[j_line][5])
                                    list_vy.append(list_PnV[j_line][6])
                                    list_vz.append(list_PnV[j_line][7])
                                    list_vt.append(list_PnV[j_line][8])
                                    # fill tau
                                    list_tau_id.append(list_PnV[k_line][3])
                                    list_tau_px.append(list_PnV[k_line][4])
                                    list_tau_py.append(list_PnV[k_line][5])
                                    list_tau_pz.append(list_PnV[k_line][6])
                                    list_tau_E.append(list_PnV[k_line][7])
                                    list_tau_mass.append(list_PnV[k_line][8])
        # prepare for the next event
        list_PnV.clear()
        #print(line)
    else:
        list_PnV.append(line_list)

print(len(list_tau_E), len(list_vx), len(list_pi0_mass))

Nevents=len(list_pi0_E) # number of pi0
Nfiles=int(Nevents / evt_per_file) +1
print(Nfiles)

###Nfiles=1

for ifile in range(Nfiles):
    f_out=open(outputfile,"w")
    line_out="HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\nW Weight\n"
    f_out.write(line_out)
    # number of events per file
    this_Nevts=evt_per_file
    if ifile==Nfiles-1: # the last file
        this_Nevts=Nevents - (Nfiles-1) * evt_per_file
    for ievt in range(this_Nevts):
        line_out="E " + str(ievt) +" 4 6\nU GEV MM\nW 1.0000000000000000000000e+00\nA 0 GenCrossSection 2.21030347e-03 5.72224450e-04 -1 -1\nA 0 GenPdfInfo 11 -11 1.00000000e+00 1.00000000e+00 2.40000000e+02 1.00000000e+00 1.00000000e+00 0 0\nA 0 alphaQCD 0.113158831200129\nA 0 alphaQED 0.00790421475742734\nA 0 event_scale 240\nA 0 mpi 1\nA 0 signal_process_id 904\nP 1 0 11 0.0000000000000000e+00 0.0000000000000000e+00 1.1999999999891199e+02 1.1999999999999999e+02 5.1099999999999995e-04 4\nP 2 1 11 0.0000000000000000e+00 0.0000000000000000e+00 1.2000000000000000e+02 1.2000000000000000e+02 0.0000000000000000e+00 21\nP 3 0 -11 0.0000000000000000e+00 0.0000000000000000e+00 -1.1999999999891199e+02 1.1999999999999999e+02 5.1099999999999995e-04 4\nP 4 3 -11 0.0000000000000000e+00 0.0000000000000000e+00 -1.2000000000000000e+02 1.2000000000000000e+02 0.0000000000000000e+00 21\nV -3 0 [2,4]\n"
        f_out.write(line_out)
        line_out="P 5 -3 "+list_tau_id[ievt+evt_per_file*ifile]+" " + list_tau_px[ievt+evt_per_file*ifile] + " " + list_tau_py[ievt+evt_per_file*ifile] + " " + list_tau_pz[ievt+evt_per_file*ifile] + " " + list_tau_E[ievt+evt_per_file*ifile] + " " + list_tau_mass[ievt+evt_per_file*ifile] + " 2\n"
        f_out.write(line_out)
        line_out="V -4 0 [5] @ " + list_vx[ievt+evt_per_file*ifile]  + " " + list_vy[ievt+evt_per_file*ifile] + " " + list_vz[ievt+evt_per_file*ifile] + " " + list_vt[ievt+evt_per_file*ifile]
        f_out.write(line_out)
        line_out="P 6 -4 22 " + list_pi0_px[ievt+evt_per_file*ifile] + " " + list_pi0_py[ievt+evt_per_file*ifile] + " " + list_pi0_pz[ievt+evt_per_file*ifile] + " " + str(list_pi0_E[ievt+evt_per_file*ifile]) + " " + list_pi0_mass[ievt+evt_per_file*ifile] + " 1\n"
        f_out.write(line_out)
    f_out.write("HepMC::Asciiv3-END_EVENT_LISTING\n")
    f_out.close()
