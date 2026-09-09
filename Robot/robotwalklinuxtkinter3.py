# simplified lx16a servo communication.. altered by Sujit Vasanth
# modified from danjperron's LX16A library https://github.com/danjperron/LX16A/blob/master/LX16A.py
# see youtube videos at https://www.youtube.com/watch?v=isUQWM0gjo4 and https://www.youtube.com/watch?v=XBxGyKV4W7U0000
from time import sleep 
from serial import Serial
import struct
import thread


serial = Serial("/dev/ttyUSB1",baudrate=115200,timeout=0.001); serial.setDTR(1)
global p, flag, b, flag2; p=1; flag=0; b=[0,0,0,0,0,0,0,0]; flag2=0; motorstate=0
n = [(6, 262, 167, 185, 381, 259, 388,0)] #define the absolute neutral position note that n[0][0] denotes the number of positions =6
n.append((600, 0,0,0,0,0,0,0)) #netral position
n.append((200, 0, 0, 80, 0, 0, -204,1)); n.append((300, 0, 0, 110, 0, 0, 70,0))#balance on one leg
n.append((200, 0,0,0,0,0,0,0))
n.append((200, 0, 0, 144, 0, 0, -87,1)); n.append((300, -1, 1, -159, -3, -1, -125,0))#balance on one leg

def gui():
    global flag, flag2, b, slider_1, slider, sliderval
    import Tkinter
    mGui=Tkinter.Tk()
    mGui.title('Robot CONTROLLER');mGui.geometry("750x326+340+140");mGui.resizable(0, 0);mGui.attributes("-topmost", True)
    mGui.wait_visibility(mGui);mGui.wm_attributes('-alpha', 0.9);mGui.update()
    sliderval=Tkinter.IntVar();sliderval.set(motorstate)
    slider_1=Tkinter.Scale(mGui,orient=Tkinter.HORIZONTAL,length=200,width=20,from_=0,to_=1,sliderlength=120,showvalue=0,label="Motors ON", variable=sliderval)
    slider_1.bind("<ButtonRelease-1>", updateValue)
    if motorstate==0: slider_1.config(label="Motors OFF")
    slider=[0]*6
    for a in range(0,6):
        slider[a]=Tkinter.Scale(mGui,orient=Tkinter.HORIZONTAL,length=300,width=20,from_=-200,to_=200, sliderlength=60, showvalue=1, label="Servo "+str(a+1))
        slider[a].bind("<ButtonRelease-1>", modeback)
        slider[a].bind('<B1-Motion>',lambda event, copy=a: updateValuea(copy))    
    button_1 = Tkinter.Button(mGui, text="READ SERVOS", command=readservos)
    button_2 = Tkinter.Button(mGui, text="<<<", command=previouspos)
    button_3 = Tkinter.Button(mGui, text=">>>", command=nextpos)
    slider_1.grid(row=0, column=0);button_1.grid(row=0, column=1)
    slider[0].grid(row=1, column=0);slider[1].grid(row=2, column=0);slider[2].grid(row=3, column=0)
    slider[3].grid(row=1, column=1);slider[4].grid(row=2, column=1);slider[5].grid(row=3, column=1)
    button_2.grid(row=0, column=3);button_3.grid(row=0, column=4)
    while True:
        if flag2==1:
            for a in range(0,6): slider[a].set(b[a])
            flag2=0
        mGui.update()
def sendPacket(packet):
    packet1=bytearray(packet);sum=0
    for a in packet1: sum=sum+a
    fullPacket = bytearray(struct.pack("<BB",0x55,0x55) + packet + struct.pack("<B",(~sum) & 0xff))
    serial.write(fullPacket); sleep(0.00002)
def sendReceivePacket(packet,receiveSize):
    t_id = packet[0];t_command = packet[2]
    serial.flushInput();serial.timeout=0.1;sendPacket(packet)
    r_packet = serial.read(receiveSize+3); return r_packet
def motorOrServo(id,motorMode,MotorSpeed):# motorMode 1=motor 2=servo
    sendPacket(struct.pack("<BBBBBh",id,7,29,motorMode,0,MotorSpeed)) 
def moveServo(id,position,rate):# Move servo 0-1000, rate(ms) 0-30000(slow)
    sendPacket(struct.pack("<BBBHH",id,7,1,position,rate))
def readPosition(id):
    rpacket = sendReceivePacket(struct.pack("<BBB",id,3,28),5)
    s = struct.unpack("<BBBBBhB",rpacket);return s[5]
def LoadUnload(id,mode):#Activate motor 0=OFF 1 =Active
    sendPacket(struct.pack("<BBBB",id,4,31,mode))
def setID(id,newid):# change the ID of servo
    sendPacket(struct.pack("<BBBB",id,4,13,newid))
def moveservos(speed,s1,s2,s3,s4,s5,s6):
    s=(s1,s2,s3,s4,s5,s6)
    for a in range(0,6): moveServo(a+1,n[0][a+1]+s[a],speed)
def readservosABS():
    m=[0]*6
    for a in range(0,6): m[a]= readPosition(a+1)
    return m
def readservos():
    m=[0]*6
    for a in range(0,6): m[a]= readPosition(a+1)-n[0][a+1]
    return m
def servoson(b):#1=on, 0=off
    for a in range(1,7):
        LoadUnload(a,b)
        sleep(0.1)
        LoadUnload(a,b) 
def updateValue(blank):
    global motorstate
    motorstate=sliderval.get()
    if motorstate==1: slider_1.config(label="Motors ON");servoson(1)
    else: slider_1.config(label="Motors OFF");servoson(0)
def updateValuea(b):
    moveServo(b+1, slider[b].get()+n[0][b+1],800)
def modeback(blank):
    global motorstate
    motorstate=0
def nextpos():
    global p; global flag; p=p+1
    if p > n[0][0]: p=1
    flag=1
def previouspos():
    global p; global flag; p=p-1
    if p < 1: p=n[0][0]
    if n[p][7]==1: p=p-1
    flag=1

thread.start_new_thread(gui,())
moveservos(n[1][0], n[1][1], n[1][2], n[1][3], n[1][4],n[1][5], n[1][6])
sleep(0.001*n[1][0])
servoson(0)
x=0
while True:
    x=x+1 
    if flag==1:
        moveservos(n[p][0], n[p][1], n[p][2], n[p][3], n[p][4],n[p][5], n[p][6])
        for x in range(0,9):
            sleep(0.0001*n[p][0])
            b=readservos()
            flag2=1
        if n[p][7]==1:
            p=p+1
            moveservos(n[p][0], n[p][1], n[p][2], n[p][3], n[p][4],n[p][5], n[p][6])
            for x in range(0,9):
                sleep(0.0001*n[p][0])
                b=readservos()
                flag2=1
        flag=0
        servoson(0)
        
    if x>1000:
        x=0
        try:
           b=readservos()
           flag2=1
        except:
            print "error"        
#serial.close()     
