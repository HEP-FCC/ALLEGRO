import os

debug = False

def executeCmd(cmd):
    print(cmd)
    if not debug:
        return os.system(cmd)

def downloadFile(url, f, outdir="../data"):
    os.makedirs(outdir, exist_ok=True)

    outfile = os.path.join(outdir, f)

    if not os.path.isfile(outfile):
        cmd = f'curl -L {url}/{f} -o {outfile}'
        executeCmd(cmd)

print("Downloading files for digitisation and reconstruction (if missing)")

url = 'http://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03'

downloadFile(url, 'capacitances_ecalBarrelFCCee_theta.root')
downloadFile(url, 'cellNoise_map_electronicsNoiseLevel_ecalB_ECalBarrelModuleThetaMerged_ecalE_ECalEndcapTurbine_hcalB_HCalBarrelReadout_hcalE_HCalEndcapReadout.root')
downloadFile(url, 'cellNoise_map_electronicsNoiseLevel_ecalB_thetamodulemerged_hcalB_thetaphi.root')
downloadFile(url, 'cellNoise_map_electronicsNoiseLevel_ecalB_thetamodulemerged.root')
downloadFile(url, 'cellNoise_map_endcapTurbine_electronicsNoiseLevel.root')
downloadFile(url, 'elecNoise_ecalBarrelFCCee_theta.root')
downloadFile(url, 'lgbm_calibration-CaloClusters.onnx')
downloadFile(url, 'lgbm_calibration-CaloTopoClusters.onnx')

downloadFile(url, 'neighbours_map_ecalB_thetamodulemerged_hcalB_thetaphi.root')
downloadFile(url, 'neighbours_map_ecalB_thetamodulemerged_hcalB_hcalEndcap_phitheta.root')
downloadFile(url, 'neighbours_map_ecalB_thetamodulemerged_ecalE_turbine_hcalB_hcalEndcap_phitheta.root')
downloadFile(url, 'neighbours_map_ecalB_thetamodulemerged.root')
downloadFile(url, 'neighbours_map_ecalE_turbine.root')
downloadFile(url, 'bdt-photonid-settings-EMBCaloTopoClusters.json')
downloadFile(url, 'bdt-photonid-settings-EMBCaloClusters.json')
downloadFile(url, 'bdt-photonid-weights-EMBCaloTopoClusters.onnx')
downloadFile(url, 'bdt-photonid-weights-EMBCaloClusters.onnx')

downloadFile(
    "http://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/IDEA",
    "DataAlgFORGEANT.root"
)