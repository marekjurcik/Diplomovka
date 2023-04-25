#výpočet sa uskutočňuje v jednotkách kpc, Myr, M_sun
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from gala.dynamics import mockstream as ms
import pandas as pd 
import matplotlib.pyplot as plt
import math


def stream_generator(name,alpha,delta,mu_ra,mu_dec,dist,v_rad,mass_of_cluster,core_concentration,end_time):
    #mu_ra proper motion in alpha direction
    #mu_dec proper motion in delta direction
    x=[]
    y=[]
    vx=[]
    vy=[]
    
    #týmto si jednoducho len určím, že používam galaktocentrické súradnice 
    coord.galactocentric_frame_defaults.set('v4.0')
    
    """
    v tejto časti si definujem potenciál, v ktorom bude vznikať stream 
    Je to klasický NEWT model galaxie, ktorý zahŕňa jadro, bulge, disk a halo.
    Dokumentácia: https://gala.adrian.pw/en/latest/api/gala.potential.potential.MilkyWayPotential.html
    """
    pot = gp.MilkyWayPotential()

    mass=mass_of_cluster*u.Msun
    b=core_concentration*u.pc
    
    c = coord.ICRS(ra=alpha*u.deg, dec=delta*u.deg, distance=dist*u.kpc,
               pm_ra_cosdec=mu_ra*u.mas/u.yr, pm_dec=mu_dec * u.mas/u.yr,
               radial_velocity=v_rad * u.km/u.s)
    
    c_gc = c.transform_to(coord.Galactocentric).cartesian
    init_of_object=gd.PhaseSpacePosition(c_gc)
     
    #potenciál guľovej hviezdokopy je Plumerv potenciál, lebo je sférická 
    potential=gp.PlummerPotential(m=mass, b=b, units=galactic)
    
    df = ms.FardalStreamDF()
    gen_stream = ms.MockStreamGenerator(df, pot, progenitor_potential=potential)
    integrate_stream, _ = gen_stream.run(init_of_object, mass,
                             dt=-0.2*u.Myr, t1=end_time*u.Myr, t2=0)
    x=(integrate_stream.x)/u.kpc
    y=(integrate_stream.y)/u.kpc
    z=(integrate_stream.z)/u.kpc
    vx=(integrate_stream.v_x)/(u.kpc/u.Myr)
    vy=(integrate_stream.v_y)/(u.kpc/u.Myr)
    vz=(integrate_stream.v_z)/(u.kpc/u.Myr)
    L_stream=integrate_stream.angular_momentum().value
    Lx_stream=L_stream[0]
    Ly_stream=L_stream[1]
    Lz_stream=L_stream[2]
    Ltot_stream=np.sqrt(Lx_stream**2+Ly_stream**2+Lz_stream**2)

    inc_stream=[]
    for i in range(0,len(x),1):
        inc=math.degrees(np.arccos(Lz_stream[i]/Ltot_stream[i]))
        inc_stream.append(inc)
    max_inc_stream=np.amax(inc_stream)
    min_inc_stream=np.amin(inc_stream)
    print("Max inc of stream: ", max_inc_stream,", Min inc: ", min_inc_stream)

    #vykreslenie grafov so súradnicami rýchlosti
    fig=integrate_stream.plot(alpha=0.1)
    plt.suptitle(f"{str(name)}", fontsize=14)
    
    #uloženie údajov do súboru
    dict={"x":x,"y":y,"z":z,"v_x":vx,"v_y":vy,"v_z":vz, "i":inc_stream}
    df=pd.DataFrame(dict)
    df.to_csv(f"{str(name)}_GC_{len(x)}.csv")
    
    #v tejto časti počítam súradnice prúdu v heliocentrických súradniciach
    stream_in_RADEC=integrate_stream.to_coord_frame(coord.ICRS)
    RA=stream_in_RADEC.ra.degree
    DEC=stream_in_RADEC.dec.degree
    radial_velocity=stream_in_RADEC.radial_velocity.to(u.km/u.s)
    dict_RADEC={"RA":RA,"DEC":DEC,"v_r":radial_velocity}
    df=pd.DataFrame(dict_RADEC)
    df.to_csv(f"{str(name)}_GC_RADEC_{len(x)}.csv")
    
    #zistenie priebehu inklinácie jednotlivých hviezd prúdu, aby som videl ako sa lýšia od inc GC
    L_stream=integrate_stream.angular_momentum().value
    
    #v tejto časti súradnice streamu onvertujem tak, aby som ich mohol použiť vov svojom modeli 
    #zistenie inklinácie GC
    L_GC=init_of_object.angular_momentum().value
    L_x=L_GC[0]
    L_y=L_GC[1]
    L_z=L_GC[2]
    L_GC_total=np.sqrt(L_x**2+L_y**2+L_z**2)
    inc_GC=math.degrees(np.arccos(L_GC[2]/L_GC_total))
    print(f"Inclination of the {str(name)}: ",inc_GC)
    
    #rotačná matica
    fraction=1/(L_GC_total*np.sqrt(L_x**2+L_y**2))
    M=[[L_z*L_x, L_z*L_y, -(L_x**2+L_y**2)],
       [-L_y*L_GC_total, L_x*L_GC_total, 0],
       [L_x*np.sqrt(L_x**2+L_y**2), L_y*np.sqrt(L_x**2+L_y**2), L_z*np.sqrt(L_x**2+L_y**2)]]
    #toto je výsledná rotačná matica
    R=np.dot(fraction,M)
    
    stream_pos={"x":x,"y":y,"z":z}
    stream_vel={"v_x":vx,"v_y":vy,"v_z":vz}
    df_pos=pd.DataFrame(stream_pos)
    df_vel=pd.DataFrame(stream_vel)
    
    pos_matrix=df_pos.to_numpy()
    vel_matrix=df_vel.to_numpy()
    
    x2=[]
    y2=[]
    z2=[]
    vx2=[]
    vy2=[]
    vz2=[]
    for j in range(0,len(x),1):
        pos_new=np.matmul(R,pos_matrix[j]) #rotácia polohových súradníc
        vel_new=np.matmul(R,vel_matrix[j]) #rotácia súradníc rýchlosti
        x2.append(pos_new[0])
        y2.append(pos_new[1])
        z2.append(pos_new[2])
        vx2.append(vel_new[0])
        vy2.append(vel_new[1])
        vz2.append(vel_new[2])
    #uloženie údajov do súboru
    dict_new={"x":x2,"y":y2,"z":z2,"v_x":vx2,"v_y":vy2,"v_z":vz2}
    df=pd.DataFrame(dict_new)
    df.to_csv(f"{str(name)}_converted_{len(x)}.csv")
    max_z=np.amax(z2)
    min_z=np.amin(z2)
    max_vel=np.amax(vz2)
    min_vel=np.amin(vz2)
    print("maximum z: ",max_z, ",Minimum z: ",min_z)
    print("maximum vz: ",max_vel, ",Minimum vz: ",min_vel)
    
    return integrate_stream

"""
pal5=stream_generator(name="pal5",alpha=229,delta=-0.124,mu_ra=-2.296,
                      mu_dec=-2.257,dist=22.9,v_rad=-58.7,
                      mass_of_cluster=2.5e4,core_concentration=4,end_time=4000)

omega_centauri=stream_generator(name="omega_centauri",alpha=201.70,delta=-47.48,
                                mu_ra=-3.25,mu_dec=-6.75,dist=5.49,v_rad=232.8,
                                mass_of_cluster=3.94*10**6,core_concentration=4,end_time=300)

NGC_7089=stream_generator(name="NGC_7089",alpha=323.36,delta=-0.82,mu_ra=3.43,
                          mu_dec=-2.16 ,dist=11.62,v_rad=3.8 ,
                          mass_of_cluster=6.48*10**5,core_concentration=4,
                          end_time=1000)


"""
NGC_6341=stream_generator(name="NGC_6341",alpha=259.28,delta=43.14,
                          mu_ra=,mu_dec,dist,v_rad,mass_of_cluster,core_concentration,end_time)
"""
NGC_2298=stream_generator(name,alpha,delta,mu_ra,mu_dec,dist,v_rad,mass_of_cluster,core_concentration,end_time)
NGC_1904=stream_generator(name,alpha,delta,mu_ra,mu_dec,dist,v_rad,mass_of_cluster,core_concentration,end_time)
NGC_1851=stream_generator(name,alpha,delta,mu_ra,mu_dec,dist,v_rad,mass_of_cluster,core_concentration,end_time)
NGC_362=stream_generator(name,alpha,delta,mu_ra,mu_dec,dist,v_rad,mass_of_cluster,core_concentration,end_time)
NGC_288=stream_generator(name,alpha,delta,mu_ra,mu_dec,dist,v_rad,mass_of_cluster,core_concentration,end_time)
"""