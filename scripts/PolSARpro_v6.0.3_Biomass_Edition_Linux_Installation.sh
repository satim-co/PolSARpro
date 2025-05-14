#!/bin/sh
# This is a custom version used for the development of pypolsarpro inside a docker container
echo ""
echo "-----------------------"
echo "Compilation"
cd Soft
chmod +x Compil_PolSARpro_Biomass_Edition_Linux.sh
./Compil_PolSARpro_Biomass_Edition_Linux.sh
cd ..
# Keep source code for development
# rm -r Soft/src
# rm Soft/Compil_PolSARpro_Biomass_Edition_Linux.sh
chmod +x -R Soft/bin/*

echo ""
echo "-----------------------"
echo "Desktop Entry Creation"
PsPFolder=$PWD
Line1="[Desktop Entry]"
Line2="Version=1.0"
Line3="Type=Application"
Line4="Name=PolSARpro_v6.0.3_Biomass_Edition"
Line5="Comment="
Line6="Exec=wish PolSARpro_v6.0.3_Biomass_Edition.tcl"
Line7a="Icon="
Line7b="/Tmp/PolSARproBio.ico"
Line7="$Line7a$PsPFolder$Line7b"
Line8a="Path="
Line8="$Line8a$PsPFolder"
Line9="Terminal=false"
Line10="StartupNotify=false"
Line11="Name[fr_FR]=PolSARpro-Bio v6.0.3"
echo $Line1 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line2 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line3 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line4 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line5 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line6 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line7 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line8 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line9 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line10 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
echo $Line11 >> PolSARpro_v6.0.3_Biomass_Edition.desktop
PsPDesktop=$(xdg-user-dir DESKTOP)
mv PolSARpro_v6.0.3_Biomass_Edition.desktop $PsPDesktop

echo ""
echo ""
echo "-----------------------"
echo "End of PolSARpro Installation"
echo "-----------------------"

