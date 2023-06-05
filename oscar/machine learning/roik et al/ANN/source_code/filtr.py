import sys

mezi_min_04 = []
mezi_min_03 = []
mezi_min_02 = []
mezi_min_01 = []

mezi_min_00 = []

mezi_plus_01 = []
mezi_plus_02 = []
mezi_plus_03 = []
mezi_plus_04 = []

#################-0.4######################
min_04_vim_90 = []
min_04_vim_80 = []
min_04_nevim_90 = []
min_04_nevim_80 = []
min_04_chyba_90 = []
min_04_chyba_80 = []
min_04_vim_50 = []
 ##
#################-0.3######################
min_03_vim_90 = []
min_03_vim_80 = []
min_03_nevim_90 = []
min_03_nevim_80 = []
min_03_chyba_90 = []
min_03_chyba_80 = []
min_03_vim_50 = []
###
#################-0.2######################
min_02_vim_90 = []
min_02_vim_80 = []
min_02_nevim_90 = []
min_02_nevim_80 = []
min_02_chyba_90 = []
min_02_chyba_80 = []
min_02_vim_50 = []
#####
#################-0.1######################
min_01_vim_90 = []
min_01_vim_80 = []
min_01_nevim_90 = []
min_01_nevim_80 = []
min_01_chyba_90 = []
min_01_chyba_80 = []
min_01_vim_50 = []

#### 
################0.0######################
min_00_vim_90 = []
min_00_vim_80 = []
min_00_nevim_90 = []
min_00_nevim_80 = []
min_00_chyba_90 = []
min_00_chyba_80 = []
min_00_vim_50 = []

####
#################0.1######################
plus_01_vim_90 = []
plus_01_vim_80 = []
plus_01_nevim_90 = []
plus_01_nevim_80 = []
plus_01_chyba_90 = []
plus_01_chyba_80 = []
plus_01_vim_50 = []
###
#################0.2######################
plus_02_vim_90 = []
plus_02_vim_80 = []
plus_02_nevim_90 = []
plus_02_nevim_80 = []
plus_02_chyba_90 = []
plus_02_chyba_80 = []
plus_02_vim_50 = []

###
#################0.3######################
plus_03_vim_90 = []
plus_03_vim_80 = []
plus_03_nevim_90 = []
plus_03_nevim_80 = []
plus_03_chyba_90 = []
plus_03_chyba_80 = []
plus_03_vim_50 = []

###
#################0.4######################
plus_04_vim_90 = []
plus_04_vim_80 = []
plus_04_nevim_90 = []
plus_04_nevim_80 = []
plus_04_chyba_90 = []
plus_04_chyba_80 = []
plus_04_vim_50 = []


####
##########################################
def filtr_plus(n1,n2,n3,n4,n5): 
    if n1 < n2 < n3:
        n4.append(n5)

kolik_min_04 = []
kolik_min_03 = []
kolik_min_02 = []
kolik_min_01 = []
kolik_min_00 = []
kolik_plus_01 = []
kolik_plus_02 = []
kolik_plus_03 = []
kolik_plus_04 = []


def counter(n2,n5,n4,kolik_min_04,kolik_min_03,kolik_min_02,kolik_min_01,kolik_min_00,kolik_plus_01,kolik_plus_02,kolik_plus_03,kolik_plus_04,min_04_vim_90,min_04_vim_80,min_04_nevim_90,min_04_nevim_80,min_04_chyba_90,min_04_chyba_80,min_04_vim_50,min_03_vim_90,min_03_vim_80,min_03_nevim_90,min_03_nevim_80,min_03_chyba_90,min_03_chyba_80,min_03_vim_50,min_02_vim_90,min_02_vim_80,min_02_nevim_90,min_02_nevim_80,min_02_chyba_90,min_02_chyba_80,min_02_vim_50,min_01_vim_90,min_01_vim_80,min_01_nevim_90,min_01_nevim_80,min_01_chyba_90,min_01_chyba_80,min_01_vim_50,min_00_vim_90,min_00_vim_80,min_00_nevim_90,min_00_nevim_80,min_00_chyba_90,min_00_chyba_80,min_00_vim_50,plus_01_vim_90,plus_01_vim_80,plus_01_nevim_90,plus_01_nevim_80,plus_01_chyba_90,plus_01_chyba_80,plus_01_vim_50,plus_02_vim_90,plus_02_vim_80,plus_02_nevim_90,plus_02_nevim_80,plus_02_chyba_90,plus_02_chyba_80,plus_02_vim_50,plus_03_vim_90,plus_03_vim_80,plus_03_nevim_90,plus_03_nevim_80,plus_03_chyba_90,plus_03_chyba_80,plus_03_vim_50,plus_04_vim_90,plus_04_vim_80,plus_04_nevim_90,plus_04_nevim_80,plus_04_chyba_90,plus_04_chyba_80,plus_04_vim_50):

##############################################
# -0.4 #
########################################
    if -0.5 < n2 < -0.4:
        kolik_min_04.append(1)
        if n5 > 0.9:
            min_04_vim_90.append(1)
        if n5 > 0.8:
            min_04_vim_80.append(1)
        if n5 < 0.1:
            min_04_chyba_90.append(1)
        if n5 < 0.2:
            min_04_chyba_80.append(1)
        if 0.1 < n5 < 0.9:
            min_04_nevim_90.append(1)
        if 0.2 < n5 < 0.8:
            min_04_nevim_80.append(1)
        if n5 > 0.95:
            min_04_vim_50.append(1)
##############################################
# -0.3 #
########################################
    if -0.4 < n2 < -0.3:
        kolik_min_03.append(1)
        if n5 > 0.9:
            min_03_vim_90.append(1)
        if n5 > 0.8:
            min_03_vim_80.append(1)
        if n5 < 0.1:
            min_03_chyba_90.append(1)
        if n5 < 0.2:
            min_03_chyba_80.append(1)
        if 0.1 < n5 < 0.9:
            min_03_nevim_90.append(1)
        if 0.2 < n5 < 0.8:
            min_03_nevim_80.append(1)
        if n5 > 0.95:
            min_03_vim_50.append(1)
##############################################
# -0.2 #
########################################
    if -0.3 < n2 < -0.2:
        kolik_min_02.append(1)
        if n5 > 0.9:
            min_02_vim_90.append(1)
        if n5 > 0.8:
            min_02_vim_80.append(1)
        if n5 < 0.1:
            min_02_chyba_90.append(1)
        if n5 < 0.2:
            min_02_chyba_80.append(1)
        if 0.1 < n5 < 0.9:
            min_02_nevim_90.append(1)
        if 0.2 < n5 < 0.8:
            min_02_nevim_80.append(1)
        if n5 > 0.95:
            min_02_vim_50.append(1)
##############################################
# -0.1 #
########################################
    if -0.2 < n2 < -0.1:
        kolik_min_01.append(1)
        if n5 > 0.9:
            min_01_vim_90.append(1)
        if n5 > 0.8:
            min_01_vim_80.append(1)
        if n5 < 0.1:
            min_01_chyba_90.append(1)
        if n5 < 0.2:
            min_01_chyba_80.append(1)
        if 0.1 < n5 < 0.9:
            min_01_nevim_90.append(1)
        if 0.2 < n5 < 0.8:
            min_01_nevim_80.append(1)
        if n5 > 0.95:
            min_01_vim_50.append(1)
 ##############################################
# 0.0 #
########################################
    if -0.1 < n2 < 0.0:
        kolik_min_00.append(1)
        if n5 > 0.9:
            min_00_vim_90.append(1)
        if n5 > 0.8:
            min_00_vim_80.append(1)
        if n5 < 0.1:
            min_00_chyba_90.append(1)
        if n5 < 0.2:
            min_00_chyba_80.append(1)
        if 0.1 < n5 < 0.9:
            min_00_nevim_90.append(1)
        if 0.2 < n5 < 0.8:
            min_00_nevim_80.append(1)
        if n5 > 0.95:
            min_00_vim_50.append(1)   
 ##############################################
# 0.1 #
########################################
    if 0.0 < n2 < 0.1:
        kolik_plus_01.append(1)
        if n4 > 0.9:
            plus_01_vim_90.append(1)
        if n4 > 0.8:
            plus_01_vim_80.append(1)
        if n4 < 0.1:
            plus_01_chyba_90.append(1)
        if n4 < 0.2:
            plus_01_chyba_80.append(1)
        if 0.1 < n4 < 0.9:
            plus_01_nevim_90.append(1)
        if 0.2 < n4 < 0.8:
            plus_01_nevim_80.append(1)
        if n4 > 0.05:
            plus_01_vim_50.append(1) 
 ##############################################
# 0.2 #
########################################
    if 0.1 < n2 < 0.2:
        kolik_plus_02.append(1)
        if n4 > 0.9:
            plus_02_vim_90.append(1)
        if n4 > 0.8:
            plus_02_vim_80.append(1)
        if n4 < 0.1:
            plus_02_chyba_90.append(1)
        if n4 < 0.2:
            plus_02_chyba_80.append(1)
        if 0.1 < n4 < 0.9:
            plus_02_nevim_90.append(1)
        if 0.2 < n4 < 0.8:
            plus_02_nevim_80.append(1)
        if n4 > 0.05:
            plus_02_vim_50.append(1)
 ##############################################
# 0.3 #
########################################
    if 0.2 < n2 < 0.3:
        kolik_plus_03.append(1)
        if n4 > 0.9:
            plus_03_vim_90.append(1)
        if n4 > 0.8:
            plus_03_vim_80.append(1)
        if n4 < 0.1:
            plus_03_chyba_90.append(1)
        if n4 < 0.2:
            plus_03_chyba_80.append(1)
        if 0.1 < n4 < 0.9:
            plus_03_nevim_90.append(1)
        if 0.2 < n4 < 0.8:
            plus_03_nevim_80.append(1)
        if n4 > 0.05:
            plus_03_vim_50.append(1)
 ##############################################
# 0.4 #
########################################
    if 0.3 < n2 < 0.4:
        kolik_plus_04.append(1)
        if n4 > 0.9:
            plus_04_vim_90.append(1)
        if n4 > 0.8:
            plus_04_vim_80.append(1)
        if n4 < 0.1:
            plus_04_chyba_90.append(1)
        if n4 < 0.2:
            plus_04_chyba_80.append(1)
        if 0.1 < n4 < 0.9:
            plus_04_nevim_90.append(1)
        if 0.2 < n4 < 0.8:
            plus_04_nevim_80.append(1)
        if n4 > 0.05:
            plus_04_vim_50.append(1)
 




def deleni(n1):
    j = len(n1)
    Sum = sum(n1)
    if j == 0:
        res = 0
    if j > 0:
        res = Sum/j   
    return res




with open ("statictika_6_1.txt","r") as data:       
    line = data.readline().rstrip()
    n=line.split()
    n = [float(x) for x in n] 
    H = len(n)
    print(n)
    min04=filtr_plus(-0.5,n[2],-0.4,mezi_min_04,n[0])
    min03=filtr_plus(-0.4,n[2],-0.3,mezi_min_03,n[0])
    min02=filtr_plus(-0.3,n[2],-0.2,mezi_min_02,n[0])
    min01=filtr_plus(-0.2,n[2],-0.1,mezi_min_01,n[0])
    min00=filtr_plus(-0.1,n[2],0.0,mezi_min_00,n[0])
    plus01=filtr_plus(0.0,n[2],0.1,mezi_plus_01,n[1])
    plus02=filtr_plus(0.1,n[2],0.2,mezi_plus_02,n[1])
    plus03=filtr_plus(0.2,n[2],0.3,mezi_plus_03,n[1])
    plus04=filtr_plus(0.3,n[2],0.4,mezi_plus_04,n[1])
    counter(n[2],n[0],n[1],kolik_min_04,kolik_min_03,kolik_min_02,kolik_min_01,kolik_min_00,kolik_plus_01,kolik_plus_02,kolik_plus_03,kolik_plus_04,min_04_vim_90,min_04_vim_80,min_04_nevim_90,min_04_nevim_80,min_04_chyba_90,min_04_chyba_80,min_04_vim_50,min_03_vim_90,min_03_vim_80,min_03_nevim_90,min_03_nevim_80,min_03_chyba_90,min_03_chyba_80,min_03_vim_50,min_02_vim_90,min_02_vim_80,min_02_nevim_90,min_02_nevim_80,min_02_chyba_90,min_02_chyba_80,min_02_vim_50,min_01_vim_90,min_01_vim_80,min_01_nevim_90,min_01_nevim_80,min_01_chyba_90,min_01_chyba_80,min_01_vim_50,min_00_vim_90,min_00_vim_80,min_00_nevim_90,min_00_nevim_80,min_00_chyba_90,min_00_chyba_80,min_00_vim_50,plus_01_vim_90,plus_01_vim_80,plus_01_nevim_90,plus_01_nevim_80,plus_01_chyba_90,plus_01_chyba_80,plus_01_vim_50,plus_02_vim_90,plus_02_vim_80,plus_02_nevim_90,plus_02_nevim_80,plus_02_chyba_90,plus_02_chyba_80,plus_02_vim_50,plus_03_vim_90,plus_03_vim_80,plus_03_nevim_90,plus_03_nevim_80,plus_03_chyba_90,plus_03_chyba_80,plus_03_vim_50,plus_04_vim_90,plus_04_vim_80,plus_04_nevim_90,plus_04_nevim_80,plus_04_chyba_90,plus_04_chyba_80,plus_04_vim_50)              
    while line:
        line = data.readline().rstrip()                               
        n=line.split()
        n = [float(x) for x in n] 
        H = len(n)
        if H == 1:
            break
        print(n)
        min04=filtr_plus(-0.5,n[2],-0.4,mezi_min_04,n[0])
        min03=filtr_plus(-0.4,n[2],-0.3,mezi_min_03,n[0])
        min02=filtr_plus(-0.3,n[2],-0.2,mezi_min_02,n[0])
        min01=filtr_plus(-0.2,n[2],-0.1,mezi_min_01,n[0])
        min00=filtr_plus(-0.1,n[2],0.0,mezi_min_00,n[0])
        plus01=filtr_plus(0.0,n[2],0.1,mezi_plus_01,n[1])
        plus02=filtr_plus(0.1,n[2],0.2,mezi_plus_02,n[1])
        plus03=filtr_plus(0.2,n[2],0.3,mezi_plus_03,n[1])
        plus04=filtr_plus(0.3,n[2],0.4,mezi_plus_04,n[1])
        counter(n[2],n[0],n[1],kolik_min_04,kolik_min_03,kolik_min_02,kolik_min_01,kolik_min_00,kolik_plus_01,kolik_plus_02,kolik_plus_03,kolik_plus_04,min_04_vim_90,min_04_vim_80,min_04_nevim_90,min_04_nevim_80,min_04_chyba_90,min_04_chyba_80,min_04_vim_50,min_03_vim_90,min_03_vim_80,min_03_nevim_90,min_03_nevim_80,min_03_chyba_90,min_03_chyba_80,min_03_vim_50,min_02_vim_90,min_02_vim_80,min_02_nevim_90,min_02_nevim_80,min_02_chyba_90,min_02_chyba_80,min_02_vim_50,min_01_vim_90,min_01_vim_80,min_01_nevim_90,min_01_nevim_80,min_01_chyba_90,min_01_chyba_80,min_01_vim_50,min_00_vim_90,min_00_vim_80,min_00_nevim_90,min_00_nevim_80,min_00_chyba_90,min_00_chyba_80,min_00_vim_50,plus_01_vim_90,plus_01_vim_80,plus_01_nevim_90,plus_01_nevim_80,plus_01_chyba_90,plus_01_chyba_80,plus_01_vim_50,plus_02_vim_90,plus_02_vim_80,plus_02_nevim_90,plus_02_nevim_80,plus_02_chyba_90,plus_02_chyba_80,plus_02_vim_50,plus_03_vim_90,plus_03_vim_80,plus_03_nevim_90,plus_03_nevim_80,plus_03_chyba_90,plus_03_chyba_80,plus_03_vim_50,plus_04_vim_90,plus_04_vim_80,plus_04_nevim_90,plus_04_nevim_80,plus_04_chyba_90,plus_04_chyba_80,plus_04_vim_50)




final_min_04 = deleni(mezi_min_04)
final_min_03 = deleni(mezi_min_03)
final_min_02 = deleni(mezi_min_02)
final_min_01 = deleni(mezi_min_01)
final_min_00 = deleni(mezi_min_00)
final_plus_01 = deleni(mezi_plus_01)
final_plus_02 = deleni(mezi_plus_02)
final_plus_03 = deleni(mezi_plus_03)
final_plus_04 = deleni(mezi_plus_04)



################ min 04#########################
Sum_min_04_vim_90 = len(min_04_vim_90)
Sum_min_04_vim_80 = len(min_04_vim_80)
Sum_min_04_nevim_90 = len(min_04_nevim_90)
Sum_min_04_nevim_80 = len(min_04_nevim_80)
Sum_min_04_chyba_90 = len(min_04_chyba_90)
Sum_min_04_chyba_80 = len(min_04_chyba_80)
Sum_min_04_vim_50 = len(min_04_vim_50)

################ min 03#########################
Sum_min_03_vim_90 = len(min_03_vim_90)
Sum_min_03_vim_80 = len(min_03_vim_80)
Sum_min_03_nevim_90 = len(min_03_nevim_90)
Sum_min_03_nevim_80 = len(min_03_nevim_80)
Sum_min_03_chyba_90 = len(min_03_chyba_90)
Sum_min_03_chyba_80 = len(min_03_chyba_80)
Sum_min_03_vim_50 = len(min_03_vim_50)

################ min 02#########################
Sum_min_02_vim_90 = len(min_02_vim_90)
Sum_min_02_vim_80 = len(min_02_vim_80)
Sum_min_02_nevim_90 = len(min_02_nevim_90)
Sum_min_02_nevim_80 = len(min_02_nevim_80)
Sum_min_02_chyba_90 = len(min_02_chyba_90)
Sum_min_02_chyba_80 = len(min_02_chyba_80)
Sum_min_02_vim_50 = len(min_02_vim_50)

################ min 01#########################
Sum_min_01_vim_90 = len(min_01_vim_90)
Sum_min_01_vim_80 = len(min_01_vim_80)
Sum_min_01_nevim_90 = len(min_01_nevim_90)
Sum_min_01_nevim_80 = len(min_01_nevim_80)
Sum_min_01_chyba_90 = len(min_01_chyba_90)
Sum_min_01_chyba_80 = len(min_01_chyba_80)
Sum_min_01_vim_50 = len(min_01_vim_50)

################ min 00#########################
Sum_min_00_vim_90 = len(min_00_vim_90)
Sum_min_00_vim_80 = len(min_00_vim_80)
Sum_min_00_nevim_90 = len(min_00_nevim_90)
Sum_min_00_nevim_80 = len(min_00_nevim_80)
Sum_min_00_chyba_90 = len(min_00_chyba_90)
Sum_min_00_chyba_80 = len(min_00_chyba_80)
Sum_min_00_vim_50 = len(min_00_vim_50)

################ plus 01#########################
Sum_plus_01_vim_90 = len(plus_01_vim_90)
Sum_plus_01_vim_80 = len(plus_01_vim_80)
Sum_plus_01_nevim_90 = len(plus_01_nevim_90)
Sum_plus_01_nevim_80 = len(plus_01_nevim_80)
Sum_plus_01_chyba_90 = len(plus_01_chyba_90)
Sum_plus_01_chyba_80 = len(plus_01_chyba_80)
Sum_plus_01_vim_50 = len(plus_01_vim_50)

################ plus 02#########################
Sum_plus_02_vim_90 = len(plus_02_vim_90)
Sum_plus_02_vim_80 = len(plus_02_vim_80)
Sum_plus_02_nevim_90 = len(plus_02_nevim_90)
Sum_plus_02_nevim_80 = len(plus_02_nevim_80)
Sum_plus_02_chyba_90 = len(plus_02_chyba_90)
Sum_plus_02_chyba_80 = len(plus_02_chyba_80)
Sum_plus_02_vim_50 = len(plus_02_vim_50)

################ plus 03#########################
Sum_plus_03_vim_90 = len(plus_03_vim_90)
Sum_plus_03_vim_80 = len(plus_03_vim_80)
Sum_plus_03_nevim_90 = len(plus_03_nevim_90)
Sum_plus_03_nevim_80 = len(plus_03_nevim_80)
Sum_plus_03_chyba_90 = len(plus_03_chyba_90)
Sum_plus_03_chyba_80 = len(plus_03_chyba_80)
Sum_plus_03_vim_50 = len(plus_03_vim_50)

################ plus 04#########################
Sum_plus_04_vim_90 = len(plus_04_vim_90)
Sum_plus_04_vim_80 = len(plus_04_vim_80)
Sum_plus_04_nevim_90 = len(plus_04_nevim_90)
Sum_plus_04_nevim_80 = len(plus_04_nevim_80)
Sum_plus_04_chyba_90 = len(plus_04_chyba_90)
Sum_plus_04_chyba_80 = len(plus_04_chyba_80)
Sum_plus_04_vim_50 = len(plus_04_vim_50)


print(sum(kolik_min_04))
print(sum(kolik_min_03))
print(sum(kolik_min_02))
print(sum(kolik_min_01))
print(sum(kolik_min_00))
print(sum(kolik_plus_01))
print(sum(kolik_plus_02))
print(sum(kolik_plus_03))
print(sum(kolik_plus_04))




with open("resoult_6_1.txt",'w') as f :
    if final_min_04 == 0:
        final_min_04 = 1
    final_min_04_txt = str(final_min_04)
    if final_min_03 == 0:
        final_min_03 = 1
    final_min_03_txt = str(final_min_03)
    if final_min_02 == 0:
        final_min_02 = 1
    final_min_02_txt = str(final_min_02)
    final_min_01_txt = str(final_min_01) 
    final_min_00_txt = str(final_min_00)
    final_plus_01_txt = str(final_plus_01)
    if final_plus_02 == 0:
        final_plus_02 = 1
    final_plus_02_txt = str(final_plus_02)
    if final_plus_03 == 0:
        final_plus_03 = 1
    final_plus_03_txt = str(final_plus_03)
    if final_plus_04 == 0:
        final_plus_04 = 1
    final_plus_04_txt = str(final_plus_04)     

    f.writelines(final_min_04_txt)
    f.write("\n")
    f.writelines(final_min_03_txt)
    f.write("\n")
    f.writelines(final_min_02_txt)
    f.write("\n")
    f.writelines(final_min_01_txt)
    f.write("\n")
    f.writelines(final_min_00_txt)
    f.write("\n")
    f.writelines(final_plus_01_txt)
    f.write("\n")
    f.writelines(final_plus_02_txt)
    f.write("\n")
    f.writelines(final_plus_03_txt)
    f.write("\n")
    f.writelines(final_plus_04_txt)
    f.write("\n")
    f.write("Kolik vzorku")
    f.write("\n")
######## kolik #######################################

    kolik_min_04_txt = str(sum(kolik_min_04))
    kolik_min_03_txt = str(sum(kolik_min_03))
    kolik_min_02_txt = str(sum(kolik_min_02))
    kolik_min_01_txt = str(sum(kolik_min_01) )
    kolik_min_00_txt = str(sum(kolik_min_00))
    kolik_plus_01_txt = str(sum(kolik_plus_01))
    kolik_plus_02_txt = str(sum(kolik_plus_02))
    kolik_plus_03_txt = str(sum(kolik_plus_03))
    kolik_plus_04_txt = str(sum(kolik_plus_04))     
    f.writelines(kolik_min_04_txt)
    f.write("\n")
    f.writelines(kolik_min_03_txt)
    f.write("\n")
    f.writelines(kolik_min_02_txt)
    f.write("\n")
    f.writelines(kolik_min_01_txt)
    f.write("\n")
    f.writelines(kolik_min_00_txt)
    f.write("\n")
    f.writelines(kolik_plus_01_txt)
    f.write("\n")
    f.writelines(kolik_plus_02_txt)
    f.write("\n")
    f.writelines(kolik_plus_03_txt)
    f.write("\n")
    f.writelines(kolik_plus_04_txt)
    f.write("\n")



    f.write("Min 04")
    f.write("\n")
######## min_04 #######################################
    min_04_vim_90_txt = str(Sum_min_04_vim_90)
    min_04_vim_80_txt = str(Sum_min_04_vim_80)
    min_04_nevim_90_txt = str(Sum_min_04_nevim_90)
    min_04_nevim_80_txt = str(Sum_min_04_nevim_80) 
    min_04_chyba_90_txt = str(Sum_min_04_chyba_90)
    min_04_chyba_80_txt = str(Sum_min_04_chyba_80)
    min_04_vim_50_txt = str(Sum_min_04_vim_50)     
    f.writelines(min_04_vim_90_txt)
    f.write("\n")
    f.writelines(min_04_vim_80_txt)
    f.write("\n")
    f.writelines(min_04_nevim_90_txt)
    f.write("\n")
    f.writelines(min_04_nevim_80_txt)
    f.write("\n")
    f.writelines(min_04_chyba_90_txt)
    f.write("\n")
    f.writelines(min_04_chyba_80_txt)
    f.write("\n")
    f.writelines(min_04_vim_50_txt)
    f.write("\n")
    f.write("Min 03")
    f.write("\n")
######## min_03 #######################################
    min_03_vim_90_txt = str(Sum_min_03_vim_90)
    min_03_vim_80_txt = str(Sum_min_03_vim_80)
    min_03_nevim_90_txt = str(Sum_min_03_nevim_90)
    min_03_nevim_80_txt = str(Sum_min_03_nevim_80) 
    min_03_chyba_90_txt = str(Sum_min_03_chyba_90)
    min_03_chyba_80_txt = str(Sum_min_03_chyba_80)
    min_03_vim_50_txt = str(Sum_min_03_vim_50)     
    f.writelines(min_03_vim_90_txt)
    f.write("\n")
    f.writelines(min_03_vim_80_txt)
    f.write("\n")
    f.writelines(min_03_nevim_90_txt)
    f.write("\n")
    f.writelines(min_03_nevim_80_txt)
    f.write("\n")
    f.writelines(min_03_chyba_90_txt)
    f.write("\n")
    f.writelines(min_03_chyba_80_txt)
    f.write("\n")
    f.writelines(min_03_vim_50_txt)
    f.write("\n")
    f.write("Min 02")
    f.write("\n")
######## min_02 #######################################
    min_02_vim_90_txt = str(Sum_min_02_vim_90)
    min_02_vim_80_txt = str(Sum_min_02_vim_80)
    min_02_nevim_90_txt = str(Sum_min_02_nevim_90)
    min_02_nevim_80_txt = str(Sum_min_02_nevim_80) 
    min_02_chyba_90_txt = str(Sum_min_02_chyba_90)
    min_02_chyba_80_txt = str(Sum_min_02_chyba_80)
    min_02_vim_50_txt = str(Sum_min_02_vim_50)     
    f.writelines(min_02_vim_90_txt)
    f.write("\n")
    f.writelines(min_02_vim_80_txt)
    f.write("\n")
    f.writelines(min_02_nevim_90_txt)
    f.write("\n")
    f.writelines(min_02_nevim_80_txt)
    f.write("\n")
    f.writelines(min_02_chyba_90_txt)
    f.write("\n")
    f.writelines(min_02_chyba_80_txt)
    f.write("\n")
    f.writelines(min_02_vim_50_txt)
    f.write("\n")
    f.write("Min 01")
    f.write("\n")
######## min_01 #######################################
    min_01_vim_90_txt = str(Sum_min_01_vim_90)
    min_01_vim_80_txt = str(Sum_min_01_vim_80)
    min_01_nevim_90_txt = str(Sum_min_01_nevim_90)
    min_01_nevim_80_txt = str(Sum_min_01_nevim_80) 
    min_01_chyba_90_txt = str(Sum_min_01_chyba_90)
    min_01_chyba_80_txt = str(Sum_min_01_chyba_80)
    min_01_vim_50_txt = str(Sum_min_01_vim_50)     
    f.writelines(min_01_vim_90_txt)
    f.write("\n")
    f.writelines(min_01_vim_80_txt)
    f.write("\n")
    f.writelines(min_01_nevim_90_txt)
    f.write("\n")
    f.writelines(min_01_nevim_80_txt)
    f.write("\n")
    f.writelines(min_01_chyba_90_txt)
    f.write("\n")
    f.writelines(min_01_chyba_80_txt)
    f.write("\n")
    f.writelines(min_01_vim_50_txt)
    f.write("\n")
    f.write("Min 00")
    f.write("\n")
######## min_00 #######################################
    min_00_vim_90_txt = str(Sum_min_00_vim_90)
    min_00_vim_80_txt = str(Sum_min_00_vim_80)
    min_00_nevim_90_txt = str(Sum_min_00_nevim_90)
    min_00_nevim_80_txt = str(Sum_min_00_nevim_80) 
    min_00_chyba_90_txt = str(Sum_min_00_chyba_90)
    min_00_chyba_80_txt = str(Sum_min_00_chyba_80)
    min_00_vim_50_txt = str(Sum_min_00_vim_50)     
    f.writelines(min_00_vim_90_txt)
    f.write("\n")
    f.writelines(min_00_vim_80_txt)
    f.write("\n")
    f.writelines(min_00_nevim_90_txt)
    f.write("\n")
    f.writelines(min_00_nevim_80_txt)
    f.write("\n")
    f.writelines(min_00_chyba_90_txt)
    f.write("\n")
    f.writelines(min_00_chyba_80_txt)
    f.write("\n")
    f.writelines(min_00_vim_50_txt)
    f.write("\n")
    f.write("Plus 01")
    f.write("\n")
######## Plus_01 #######################################
    plus_01_vim_90_txt = str(Sum_plus_01_vim_90)
    plus_01_vim_80_txt = str(Sum_plus_01_vim_80)
    plus_01_nevim_90_txt = str(Sum_plus_01_nevim_90)
    plus_01_nevim_80_txt = str(Sum_plus_01_nevim_80) 
    plus_01_chyba_90_txt = str(Sum_plus_01_chyba_90)
    plus_01_chyba_80_txt = str(Sum_plus_01_chyba_80)
    plus_01_vim_50_txt = str(Sum_plus_01_vim_50)     
    f.writelines(plus_01_vim_90_txt)
    f.write("\n")
    f.writelines(plus_01_vim_80_txt)
    f.write("\n")
    f.writelines(plus_01_nevim_90_txt)
    f.write("\n")
    f.writelines(plus_01_nevim_80_txt)
    f.write("\n")
    f.writelines(plus_01_chyba_90_txt)
    f.write("\n")
    f.writelines(plus_01_chyba_80_txt)
    f.write("\n")
    f.writelines(plus_01_vim_50_txt)
    f.write("\n")
    f.write("Plus 02")
    f.write("\n")
######## Plus_02 #######################################
    plus_02_vim_90_txt = str(Sum_plus_02_vim_90)
    plus_02_vim_80_txt = str(Sum_plus_02_vim_80)
    plus_02_nevim_90_txt = str(Sum_plus_02_nevim_90)
    plus_02_nevim_80_txt = str(Sum_plus_02_nevim_80) 
    plus_02_chyba_90_txt = str(Sum_plus_02_chyba_90)
    plus_02_chyba_80_txt = str(Sum_plus_02_chyba_80)
    plus_02_vim_50_txt = str(Sum_plus_02_vim_50)     
    f.writelines(plus_02_vim_90_txt)
    f.write("\n")
    f.writelines(plus_02_vim_80_txt)
    f.write("\n")
    f.writelines(plus_02_nevim_90_txt)
    f.write("\n")
    f.writelines(plus_02_nevim_80_txt)
    f.write("\n")
    f.writelines(plus_02_chyba_90_txt)
    f.write("\n")
    f.writelines(plus_02_chyba_80_txt)
    f.write("\n")
    f.writelines(plus_02_vim_50_txt)
    f.write("\n")
    f.write("Plus 03")
    f.write("\n")
######## Plus_03 #######################################
    plus_03_vim_90_txt = str(Sum_plus_03_vim_90)
    plus_03_vim_80_txt = str(Sum_plus_03_vim_80)
    plus_03_nevim_90_txt = str(Sum_plus_03_nevim_90)
    plus_03_nevim_80_txt = str(Sum_plus_03_nevim_80) 
    plus_03_chyba_90_txt = str(Sum_plus_03_chyba_90)
    plus_03_chyba_80_txt = str(Sum_plus_03_chyba_80)
    plus_03_vim_50_txt = str(Sum_plus_03_vim_50)     
    f.writelines(plus_03_vim_90_txt)
    f.write("\n")
    f.writelines(plus_03_vim_80_txt)
    f.write("\n")
    f.writelines(plus_03_nevim_90_txt)
    f.write("\n")
    f.writelines(plus_03_nevim_80_txt)
    f.write("\n")
    f.writelines(plus_03_chyba_90_txt)
    f.write("\n")
    f.writelines(plus_03_chyba_80_txt)
    f.write("\n")
    f.writelines(plus_03_vim_50_txt)
    f.write("\n")
    f.write("Plus 04")
    f.write("\n")
######## Plus_04 #######################################
    plus_04_vim_90_txt = str(Sum_plus_04_vim_90)
    plus_04_vim_80_txt = str(Sum_plus_04_vim_80)
    plus_04_nevim_90_txt = str(Sum_plus_04_nevim_90)
    plus_04_nevim_80_txt = str(Sum_plus_04_nevim_80) 
    plus_04_chyba_90_txt = str(Sum_plus_04_chyba_90)
    plus_04_chyba_80_txt = str(Sum_plus_04_chyba_80)
    plus_04_vim_50_txt = str(Sum_plus_04_vim_50)     
    f.writelines(plus_04_vim_90_txt)
    f.write("\n")
    f.writelines(plus_04_vim_80_txt)
    f.write("\n")
    f.writelines(plus_04_nevim_90_txt)
    f.write("\n")
    f.writelines(plus_04_nevim_80_txt)
    f.write("\n")
    f.writelines(plus_04_chyba_90_txt)
    f.write("\n")
    f.writelines(plus_04_chyba_80_txt)
    f.write("\n")
    f.writelines(plus_04_vim_50_txt)
    f.write("\n")


