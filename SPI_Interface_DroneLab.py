import spidev
import RPi.GPIO as GPIO
import math
import time
import numpy as np
from threading import Thread
import datetime
from tqdm import tqdm

def flatten(xss):
    return [x for xs in xss for x in xs]

def convertBinaryList(binList):
    return [int(i, 2) for i in binList]

def listsplit(a:list, n:int):
    k, m = divmod(len(a), n)
    return (list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))

class crazy_processor():
    
    def __init__(
        self,
        instruction_file:str,
        parameter_file:str = None,
        spi_bus:int = 0, 
        spi_device:int = 1, 
        spi_speed:int = 3000000,
        spi_mode:int = 0b00,
        interrupt_pin:int = 25,
        packet_pin:int = 24,
        reset_pin:int = 5
        ):
        self._bus = spi_bus
        self._device = spi_device
        self._spi = spidev.SpiDev()
        self._spi_speed = spi_speed
        self._spi_mode = spi_mode
        self._interrupt_pin = interrupt_pin
        self._packet_pin = packet_pin
        self.reset_pin = reset_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._interrupt_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self._packet_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.reset_pin, GPIO.OUT)
        GPIO.output(self.reset_pin, GPIO.HIGH)
        GPIO.add_event_detect(self._packet_pin, GPIO.RISING, callback=self._packet_callback)
        self.processor_reset()

        self._packet_count:int = 0
        #self._instruction_packet_count:int = 0
        #self._parameter_packet_count:int = 0
        #self._activation_packet_count:int = 0        
        #time.sleep(10)
        self.processor_init(instruction_file, parameter_file)
            
    def spi_decorator(func):
        def wrapper(self, *args,**kwargs):
            self._spi.open(self._bus, self._device)
            self._spi.max_speed_hz = self._spi_speed
            self._spi.mode = self._spi_mode
            out = func(self, *args,**kwargs)
            self._spi.close()
            return out
        return wrapper

    def memoryPreprocessing(self, memoryFile:str):
        """
        Prepare memory file to be converted into spi packet
        Takes the string data from the memory file and converts it to integers.
        """
        with open(memoryFile, "r") as f:
            data = f.read()
        
        dataList = data.split('\n')
        dataList = list(filter(None, dataList))
        integerList = [int(i, 2) for i in dataList]
        return integerList


    def packetGenerator(self, data:list[int], packetType:str, read:bool = False, startAddress = None, messageSize = None, packetByteSize:int = 350):
        """
        Generate SPI packets with appropriate headers
        """
        # Set up header components and memory information
        if packetType == "instruction":
            memoryHeader = 192 #11
            addressWidth = 6
            dataWidth = 80
        elif packetType == "parameter":
            memoryHeader = 64 #01
            addressWidth = 15
            dataWidth = 128
        elif packetType == "activation":
            memoryHeader = 128 #10
            addressWidth = 12
            dataWidth = 8

        if read:
            # Set up read header component
            readHeader = 32 #1
            # Set up read blank data
            dataList = [0]*messageSize
        else:
            # Set up write header component
            readHeader = 0 #0
            # Prepare write data
            dataList = data

        # Size of data that is going to be transferred
        dataPacketSize = len(dataList)-1
        # Number of bytes in a complete word
        numBytes = (dataWidth)//8
        # Number of words that can be transferred in a single SPI packet based on limits set by packetByteSize MAX = 4096
        numWords = packetByteSize//numBytes
        # Number of packets the data needs to split into to send all data to processor
        numPackets = math.ceil(dataPacketSize/numWords)
        # List of packets split by number of packets required to send
        SPI_Packet_list = listsplit(dataList, numPackets)

        # Burst header information set up
        outputDataList = []
        if startAddress is None: # Default start address to 0
            startAddress = 0
        spi_packet_address = startAddress
        packetSize = 0
        for packetNum, packet in enumerate(SPI_Packet_list):
            
            spi_packet_address += packetSize
            packetSize = len(packet)
            # Creating Burst header information
            burstTop = (packetSize >> 16) & 0xff
            burstMiddle = (packetSize >> 8) & 0xff
            burstBottom = (packetSize & 0xff) -1
            if burstTop:
                burstHeader = 16 # 1
                burstCount = 3
                burst = [burstTop,burstMiddle,burstBottom]
            elif ~burstTop and burstMiddle:
                burstHeader = 16 # 1
                burstCount = 2
                burst = [burstMiddle,burstBottom]
            elif ~burstTop and ~burstMiddle and burstBottom:
                burstHeader = 16 # 1
                burstCount = 1
                burst = [burstBottom]
            else:
                burstHeader = 0 # 1
                burstCount = 0
                burst = []

            # Header information Set up
            headerIdentifier = 12

            header = memoryHeader + readHeader + burstHeader + headerIdentifier + burstCount

            # Address Information Set up
            if packetType == "instruction":
                address = [spi_packet_address]
            else:
                addressTop = (spi_packet_address >> 8) & 0xff
                addressBottom = spi_packet_address & 0xff
                address = [addressTop, addressBottom]

            # Data Information Setup
            if read:
                outputData = [0]*(packetSize+1)
            else:
                outputData = []
                for word in packet:
                    integerList = []
                    for _ in range(int(dataWidth/8)):
                        # Extract the least significant 8 bits using a bitwise AND with 0xFF (binary 11111111)
                        byte = word & 0xFF
                        integerList.append(byte)
                        # Right-shift the original integer by 8 bits to prepare for the next extraction
                        word >>= 8
                    integerList.reverse() 
                    outputData.extend(integerList)

            spiOut = [header] + burst + address + outputData
            outputDataList.append(spiOut)
                    
        return outputDataList, numPackets

    
    @spi_decorator
    def processor_init(self, instruction_file, parameter_file):
        instruction_packet_count = 0
        parameter_packet_count = 0

        
        instructionData = self.memoryPreprocessing(instruction_file)
        spiData, numInstPackets = self.packetGenerator(instructionData, "instruction")
        self._packet_count = 0
        for packet in spiData:
            resp = self._spi.xfer2(packet)
        instruction_packet_count = self._packet_count

        parameterData = self.memoryPreprocessing(parameter_file)
        spiData, numParamPackets = self.packetGenerator(parameterData, "parameter")
        self._packet_count = 0
        for packet in spiData:
            resp = self._spi.xfer2(packet)
        parameter_packet_count = self._packet_count
        
        if parameter_packet_count != numParamPackets or instruction_packet_count != numInstPackets:
            paramFail = parameter_packet_count != numParamPackets
            instFail = instruction_packet_count != numInstPackets
            self.processor_hard_reset()
            #raise Exception(f"Failed to send {'Parameters' if paramFail else 'Instructions'}\n Expected {numParamPackets if paramFail else numInstPackets} packets, got {parameter_packet_count if paramFail else instruction_packet_count}")
        else:
            print("Processor instructions and parameters loaded successfully.")

    def processor_hard_reset(self):
        GPIO.output(self.reset_pin, GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(self.reset_pin, GPIO.HIGH)
    
    @spi_decorator
    def processor_reset(self): 
        resp = self._spi.xfer2([int("00001101", 2)])

    @spi_decorator
    def processor_enable(self):
        resp = self._spi.xfer2([int("00001110", 2)])

    @spi_decorator
    def processor_disable(self):
        resp = self._spi.xfer2([int("00001100", 2)])

    @spi_decorator
    def _send_activation(self, data):
        activation_packet_count = 0
        self._packet_count = 0
        spiData, numPackets = self.packetGenerator(data, "activation")
        
        for packet in spiData:
            #print(packet)
            resp = self._spi.xfer2(packet)  
        
        activation_packet_count = self._packet_count
        if numPackets != activation_packet_count:
            pass
            #print("Packet Transmission Failed")
            #self.processor_hard_reset()

    @spi_decorator
    def _read_output(self, startAddress, outputLength):
        self._packet_count = 0
        spiData, numPackets = self.packetGenerator([], "activation", True,startAddress, outputLength)
        outputData = []
        
        for packet in spiData:
            resp = self._spi.xfer2(packet)
            outputData.extend(resp[-outputLength:])
        #print(self._packet_count)
        return outputData
    
    
    def _triggerWait(self):     
        GPIO.setup(self._interrupt_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        ret = GPIO.wait_for_edge(self._interrupt_pin, GPIO.RISING, timeout=20)
        if ret is None:
            while True:
                pin_state = GPIO.input(self._interrupt_pin)
                if pin_state == GPIO.HIGH:
                    break
                else:
                    pass
    
    def _packet_callback(self,arg1):
        self._packet_count += 1

    def instruction_packet(self):
        return self._instruction_packet_count 
    
    def param_packet(self):
        return self._parameter_packet_count 
    
    def activation_packet(self):
        return self._activation_packet_count 
    
    def processor_run(self, activation_data, outputLength:int = 4, startAddress:int = 0):
        triggerWait = Thread(target=self._triggerWait, daemon=True)
        triggerWait.start()
        #print(str(datetime.datetime.now().strftime("%M:%S:%f")), "Start activation")
        self._activation_packet_count = self._send_activation(activation_data)
        #time.sleep(10)
        self.processor_reset()
        #time.sleep(10)
        self.processor_enable()

        
        triggerWait.join()
        #time.sleep(10)
        #print(str(datetime.datetime.now().strftime("%M:%S:%f")), "Read output")
        outputData = self._read_output(startAddress,outputLength)
        #time.sleep(0.01)
        #time.sleep(10)
        self.processor_disable()
        #time.sleep(10)
        self.processor_reset()
        return outputData
                

if __name__ == "__main__":
#########################################################################################################################################
    # SET THESE TO THE CORRECT FILES FOR THE PROCESS YOU WANT TO RUN
    instructionMemory = "./robodog_instruction_data.txt"
    parameterMemory = "./robodog_parameter_data.txt"
    activationMemory = "./robodog_activation_data.txt"
    startAddress = 233
    outputLength = 8

    # model_base_dir = "FixedScaleIO3_5_withStats"
    # instructionMemory = "Software/FixedScaleIO3_5_withStats/Binaries/instr_mem_bin.txt"
    # parameterMemory = "Software/FixedScaleIO3_5_withStats/Binaries/param_mem_bin.txt"
    # startAddress = 256
    # outputLength = 4

#########################################################################################################################################
    # Initialise Processor
    crazy_proc = crazy_processor(instructionMemory, parameterMemory)
#########################################################################################################################################
    # Prepare Activation Data

    activationData = crazy_proc.memoryPreprocessing(activationMemory)

    # pre_data = [0.046159707-0,	-0.043740854-0.5,	0.5390565395355225-0.5,	0.003083089,	0.003351754,	-0.002525565]
    # pre_data = [int(round((x / 0.0236),0)) for x in pre_data]
    # activationData = [pre_data[0], pre_data[1], pre_data[2],    # dx - tx, dy - ty, dz - tz
    #         pre_data[3], pre_data[4], pre_data[5],    # vx, vy, vz
    #         -211, -211,-21,0,0,-20,
    #         -220, -211,-21,0,0,-20,
    #         -229, -211,-21,0,0,-20,
    #         -238, -211,-21,0,0,-20,
    #         -247, -211,-21,0,0,-20,
    #         -245, -211,-21,0,0,-20,
    #         -254, -211,-21,0,0,-20,
    #         -262, -211,-21,0,0,-20]
    # for x in range(len(activationData)):
    #     if activationData[x] > 125: activationData[x]=125
    #     elif activationData[x] < -125: activationData[x]=-125        
             
    # bin_data = []
    # for num in activationData:
    #     bin_num = bin(num)[2:].zfill(8) if num >= 0 else bin(num+256)[2:].zfill(8)
    #     bin_data.append(bin_num)

#########################################################################################################################################
    # Run Processor
    starttime = time.monotonic_ns()
    outputData = crazy_proc.processor_run(activationData, startAddress=startAddress, outputLength=outputLength)
    endtime = time.monotonic_ns()
#########################################################################################################################################
    # Post Processing
    integerOutput = []
    for i in range(len(outputData)):
        if outputData[i] > 127:
            integerOutput.append(outputData[i] - 256)
        else:
            integerOutput.append(outputData[i])

    timeTaken = (endtime - starttime)/1000000000
    print("############################\nInitial Value Test\n############################")
    print("Direct FPGA Output")
    print(outputData)
    print("Signed Output Bytes:")
    print(integerOutput)
    print("Drone Input:")
    fltOutput = [x * 0.0394 for x in integerOutput]
    print(fltOutput)

    print("Processing time")
    print(f"{timeTaken} Seconds")

#########################################################################################################################################
    # Longevity Testing
    print("############################\nLong Term Transmission Test\n############################")
    for i in tqdm(range(20000)):
        outputTest = crazy_proc.processor_run(activationData, startAddress=startAddress, outputLength=outputLength)
        if outputTest != outputData:
            print(f"Output mismatch from initial test| Got {outputTest} | Expected {outputData} |")

