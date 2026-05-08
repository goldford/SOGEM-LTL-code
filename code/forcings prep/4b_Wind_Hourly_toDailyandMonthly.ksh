#!/bin/ksh
source $LMOD_PKG/init/ksh
module load cdo/1.9.8 ;

pathin="../DATA/wind_hourly/"
pathout1="../DATA/wind_daily/"
pathout2="../DATA/wind_monthly/"
pathtemp="../../data_temp/"
n=0
N1=0

echo $pathout1 ;
dir="${pathout1}" ;
if [[ ! -e $dir ]]; then
  mkdir $dir
fi

echo $pathout2 ;
dir="${pathout2}" ;
if [[ ! -e $dir ]]; then
  mkdir $dir
fi

for y in {2009..2018} ; do
  echo y: $y ;
  cdo selyear,$y ${pathin}RDRS21_NEMOgrid_wind_${y}.nc ${pathtemp}RDRS21_NEMOgrid_wind_fix_${y}.nc #fix for weird dates GO - 20230510
  cdo daymean ${pathtemp}RDRS21_NEMOgrid_wind_fix_${y}.nc ${pathout1}RDRS21_NEMOgrid_wind_daily_${y}.nc;
done


for y in {2009..2018} ; do
  echo y: $y ;
  cdo monmean ${pathout1}RDRS21_NEMOgrid_wind_daily_${y}.nc ${pathout2}RDRS21_NEMOgrid_wind_monthly_${y}.nc;

# if issues with nan or nodata or weird values look at cdo monavg
done 


# old code bleow
#for y in {1979..1983} ; do
#  echo y: $y ; 
#  echo N1: $N1 ;
#  if [[ $n -eq 1 ]] ; then
#    N2=$((N1 + 366 * 24 - 1)) 
#else
#    N2=$((N1 + 365 * 24 - 1))
#  fi
  
#  echo N2: $N2 ;
  
#  ncks -d time,$N1,$N2 ${pathin}meansspressure_2016to2020.nc ${pathout}meansspressure_${y}.nc ;
#  ncks --mk_rec_dmn time ${pathout}meansspressure_${y}.nc -o ${pathout}meansspressure_y${y}.nc ;

#  let n++; 
#  if [[ $n -eq 4 ]] ; then
#    n=0 ;
#  fi
#  N1=$((N2+1))
#  echo n: $n ;