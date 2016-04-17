from scipy.integrate import quad
import scipy
import matplotlib.pyplot as plt
class evaporator():
    
#falling evaporator design
#
    
#DATA
    ts= input("Please enter saturation temperatur in K : ")#saturation temperature in K
    t= input("Please enter surface temperature in K : ")#surface temperature in K
    d= input("Please enter Diameter of evaporator in m : ")#diameter of evaporator in m
    h= input("Please enter Height of Evaporator in m : ")#height of evaporator in m
    k=0.091#liquid conductivity in W/(m.K)
    pl=585.0#liquid density in Kg/m3
    pg=7.0#gas density in Kg/m3
    Hl=776900#latent heat of vapourzation in J/Kg
    n=0.0001589#liquid viscosity in Pa.s    
    f=0.001#liquid flow rate in Kg/s
    g=9.81#acceleration due to gravity
    A=((pl*(pl-pg)*g)/(n*n))**0.3333#useful to calculate the value of this constant as it is repeatedly used in the design
    B=(5.87*(t-ts)*k)/(Hl*n)#useful to calculate the value of this constant as it is repeatedly used in the design
    
#INLET VALUES CALCULATIONS
        
    To = f/d#To is the flow rate of liquid per unit of tube periphery at the inlet
    Ri = (4*To)/n #Ri is the Reynolds Number at the inlet
    eo = (0.75*(n*n*Ri)/(pl*(pl-pg)*g))**0.33#eo is the boundary layer thickness at the inlet
    
#SOLVING RHS OF ODE TO CALCULATE REYNOLDS NUMBER AT THE OUTLET
    
    def integrand(self):
        return 1.0
    I = quad(integrand,0,h)#I is the integration of length over small interval
    
#OUTLET VALUES CALCULATIONS
    
#IF THE FLOW IS IN LAMINAR REGIME i.e. Ro < 30
    
    if (Ri <= 30):
        C =  A * B * I[0]#Constant
        R = (Ri**1.333 - C)#R is Reynolds number at the outlet raised to 4/3
        Ro=R**0.75#Ro is the Reynolds number at the outlet
        
#SOLVING RHS OF ODE TO CALCULATE MEAN HEAT TRANSFER COEEFICIENT
  
        def integrand_1(x):
            return x**0.3333
            
        I1 = quad(integrand_1,Ro,Ri)#Valuable constant
        e = (0.75*(n*n*Ro)/(pl*(pl-pg)*g))**0.33#e is the boundary layer thickness at the outlet
        T = n*Ro/4#T is the flow rate of liquid per unit of tube periphery at the outlet
        f = (Ri - Ro)/Ri#fraction of liquid evaporated
        u1=A*1.47*(Ri-Ro)*k/I1[0]#Mean heat transfer coefficient 
        u2=(n*(Ri-Ro)*Hl)/(4*h*(t-ts))#Mean heat transfer coefficient from overall heat balance

#IF THE FLOW IS IN WAVY-LAMINAR REGIME i.e 30 < Ro < 1800        
        
    elif (Ri > 30 & Ri < 1800):
        R = (Ri**1.22)-3.69*B*h*A/(5.87)#R is Reynolds number at the outlet raised to 1.22
        Ro = R**0.8197#Ro is the Reynolds number at the outlet
        
#SOLVING RHS OF ODE TO CALCULATE MEAN HEAT TRANSFER COEEFICIENT
        
        def eva_3(x):
            return x**0.22
            
        I2 = quad(eva_3,Ro,Ri)#Valuable constant
        e = (0.75*(n*n*Ro)/(pl*(pl-pg)*g))**0.33#e is the boundary layer thickness at the outlet
        T = n*Ro/4#T is the flow rate of liquid per unit of tube periphery at the outlet
        f = (Ri - Ro)/Ri#fraction of liquid evaporated
        u1 = k*A*(Ri-Ro)*0.756/(I2[0])#Mean heat transfer coefficient
        u2=(n*(Ri-Ro)*Hl)/(4*h*(t-ts))#Mean heat transfer coefficient from overall heat balance
        
#IF THEFLOW IS IN TURBULENT REGIME i.e. Ro > 1800
        
    else:
        print ("THE FLOW IS IN TURBULENT REGIME SO THE CALCULATION IS VERY COMPLEX AND SOLUTION CANNOT BE OBTAINED USING SIMPLE PYTHON FUNCTIONS")

#TO PLOT REYNOLDS NUMBER VS LENGTH FROM THE TOP PLOT
    j=range(10)
    for i in range (9):
        h = 0.1 -  h*i/10
        C =  A * B * h#Constant
        R = (Ri**1.333 - C)#R is Reynolds number at the outlet raised to 4/3
        j[i]=R**0.75#Ro is the Reynolds number at the outlet
       
        plt.plot(j[i],h,'ro')
        h = 0.1

#PRINTING RESULTS
    print "                                                                  "
    print "-------------------------REULTS-----------------------------------"
    print "                                                                  "
    print  "1. REYNOLDS NUMBER OF THE FILM AT THE INLET = ", Ri
    print  "2. FLOW RATE AT THE TUBE INLET = " , To ," Kg/s"
    print  "3. BOUNDARY LAYER OF THE FILM AT THE INLET = ", eo , " m"
    print "                                                                  "
    print "-----------------------------------------------------------------"
    print "                                                                  "
    print  "4. REYNOLDS NUMBER OF THE FILM AT THE OUTLET = ", Ro
    print  "5. FLOW RATE AT THE TUBE OUTLET = " , T , " Kg/s"
    print  "6. BOUNDARY LAYER OF THE FILM AT THE OUTLET = ", e , " m"
    print "                                                                  "
    print  "----------------------------------------------------------------"
    print "                                                                  "
    print "7. MEAN HEAT TRANSFER COEFFICIENT = ",u1 , " W/(m2.K)"
    print "8. MEAN HEAT TRANSFER COEFFICIENT = ",u2," W/(m2.K)"
    print "   (FROM OVERALL HEAT BALANCE)"
    print "                                                                  "
    print "                                                                  " 
    print "----------------------THANK YOU----------------------------------"
    
plt.plot()
plt.show()      
        
#END OF THE DESIGN
    
    
    
    
    
  
    
    
        
    
        
        
   
        
        
        
        