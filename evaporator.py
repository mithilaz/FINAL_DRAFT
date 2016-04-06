#falling film evaporator design
ts=370.0#saturation temperature in K
d=1.0#diameter in m
l=0.1#height of evaporator in m
t=373#surface temperature in K
f=0.001#liquid flow rate in Kg/s
k=0.091#liquid conductivity in W/(m.K)
pl=585.0#liquid density in Kg/m3
pg=7.0#gas density in Kg/m3
h=776900#latent heat of evaporation in J/Kg
e=0.0#final value of boundary layer
eo=0.0#initial boundary layer value
n=0.0001589#liquid viscosity in N.s/m2.[Kg/m.s]
To=f/d#flow rate per unit of tube periphery
Ro=4*To/n#Reynolds number at inlet
a=4*n*k*(t-ts)*l
b=pl*(pl-pg)*9.816*h
c=a/b
eo=(0.75*(n*n*Ro/(pl*(pl-pg)*9.816))**0.33
#to calculate reynolds number at the outlet




