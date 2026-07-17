import json
import sys
import os

json_filename = sys.argv[1]

with open(json_filename) as f:
    data = json.load(f)

up_a = up_b = up_c = up_d = up_e = up_f = ""
down_a = down_b = down_c = down_d = down_e = down_f = ""

upstream_params_str = ""
for item in data['corr_params']:
    if item['type'] == 'upstream':
        if item['name'] == 'a':
            up_a = str(item['value'])
        if item['name'] == 'b':
            up_b = str(item['value'])
        if item['name'] == 'c':
            up_c = str(item['value'])
        if item['name'] == 'd':
            up_d = str(item['value'])
        if item['name'] == 'e':
            up_e = str(item['value'])
        if item['name'] == 'f':
            up_f = str(item['value'])
    if item['type'] == 'downstream':
        if item['name'] == 'a':
            down_a = str(item['value'])
        if item['name'] == 'b':
            down_b = str(item['value'])
        if item['name'] == 'c':
            down_c = str(item['value'])
        if item['name'] == 'd':
            down_d = str(item['value'])
        if item['name'] == 'e':
            down_e = str(item['value'])
        if item['name'] == 'f':
            down_f = str(item['value'])

upstream_params_str = up_a + ", " + up_b + ", " + up_c + ", " + up_d + ", " + up_e + ", " + up_f
downstream_params_str = down_a + ", " + down_b + ", " + down_c + ", " + down_d + ", " + down_e + ", " + down_f

cmd = "sed -i --follow-symlinks 's/ecalBarrelUpstreamParameters *=.*/ecalBarrelUpstreamParameters = [[%s]]/' run_ALLEGRO_reco.py" % upstream_params_str
print(cmd)
os.system(cmd)
cmd = "sed -i --follow-symlinks 's/ecalBarrelDownstreamParameters *=.*/ecalBarrelDownstreamParameters = [[%s]]/' run_ALLEGRO_reco.py" % downstream_params_str
print(cmd)
os.system(cmd)
