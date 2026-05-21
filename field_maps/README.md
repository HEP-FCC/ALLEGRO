Convert the txt file from COMSOL to a ROOT field map for DD4hep with
```
python convert_fieldmap.py ALLEGRO\,\ 2T\ solenoid\,\ 24-4-26.txt ALLEGRO_fieldmap_2T_20260424.root
```

In k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ECalBarrel_thetamodulemerged.xml, enable the field map replacing the constant solenoid field with
```
    <field name="SolenoidMap" type="FieldBrBz"
	   filename="ALLEGRO_fieldmap_2T_20260424.root"
	   treeName="fieldmap"
	   rhoVarName="r"
	   zVarName="z"
	   BrhoVarName="Br"
	   BzVarName="Bz"
           rScale = "1.0"
           zScale = "1.0"
           bScale = "1.0"
	   coorUnits="m"
	   BfieldUnits="tesla"
	   >
    </field>
```

Draw the B field map:
```
python drawBField.py
```
