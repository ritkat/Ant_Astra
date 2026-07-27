#!/usr/bin/env python3
"""
astra_interface.py
High-level Python interface for the Astra STM32 firmware.

The MCU boots into STATE_DISABLED.  This class tracks state transitions
by matching the printf messages the firmware emits on UART2.

Example:
    with AstraInterface("COM3", 115200) as astra:
        astra.write_paramcfg("params.bin")
        astra.write_instcfg("model.bin")
        outputs = astra.run_activation([0x01, 0x02, 0x03, 0xC0, 0xDB])
"""

import struct
import time
import serial


SLIP_END = 0xC0
SLIP_ESC = 0xDB
SLIP_ESC_END = 0xDC
SLIP_ESC_ESC = 0xDD


class AstraInterface:
    """Stateful wrapper around the Astra UART / SLIP protocol."""

    STATE_UNKNOWN = "unknown"
    STATE_DISABLED = "disabled"
    STATE_STARTUP = "startup"
    STATE_IDLE = "idle"
    STATE_SEND_ACTIVATION = "send_activation"
    STATE_READ_OUTPUT = "read_output"
    STATE_CONFIG = "config"

    stateDict = {
        0: STATE_IDLE,
        1: STATE_STARTUP,
        2: STATE_SEND_ACTIVATION,
        3: STATE_READ_OUTPUT,
        4: STATE_DISABLED,
        5: STATE_CONFIG
    }

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self._state = self.STATE_UNKNOWN
        self._rx_buf = ""

        # dsrdtr=False avoids resetting boards where DTR is tied to NRST.
        self.ser = serial.Serial(port, baud, timeout=0.005, dsrdtr=False)

        # The MCU prints its boot banner immediately on reset, long before
        # this script connects.  Wait a moment for the port to settle and
        # for the MCU to finish booting into STATE_DISABLED, then discard
        # any stale bytes.
        time.sleep(0.5)
        self.ser.reset_input_buffer()
        
        self.reset()
        self.mode()
        if self._state != self.STATE_IDLE:
            # Auto-enable the processor as requested.
            self.enable()

    def close(self):
        """Close the serial port."""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Low-level serial helpers
    # ------------------------------------------------------------------
    def _drain(self, duration: float = 0.2):
        """Read and discard anything in the RX buffer for a short time."""
        deadline = time.time() + duration
        while time.time() < deadline:
            self.ser.read(self.ser.in_waiting or 1)

    def _readlines(self, timeout: float = 0.2) -> list[str]:
        """Return any complete lines received from the MCU."""
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:

            #print(self.ser.in_waiting)
            raw = self.ser.read(max(1, self.ser.in_waiting))
            #raw = self.ser.read(self.ser.in_waiting)
            #print(raw)
            
            if raw:
                self._rx_buf += raw.decode("ascii", errors="ignore")
            while "\n" in self._rx_buf:
                line, self._rx_buf = self._rx_buf.split("\n", 1)
                line = line.strip("\r")
                if line:
                    print(line)
                    lines.append(line)
            if not raw and lines:
                break
            if not raw:
                time.sleep(0.005)
        return lines

    def _wait_for(self, *markers: str, timeout: float = 5.0) -> str:
        """
        Block until one of *markers* appears in a received line.
        Returns the line that matched.  Raises TimeoutError on expiry.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            for line in self._readlines(timeout=0.1):
                self._update_state_from_line(line)
                for marker in markers:
                    if marker in line:
                        return line
        raise TimeoutError(f"Timeout waiting for: {markers}")

    def _send_text_cmd(self, cmd: str):
        """Send a text command (automatically appends \\r\\n)."""
        self.ser.write(f"{cmd}\r\n".encode())
        #time.sleep(0.1)
        self.ser.flush()

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------
    def _update_state_from_line(self, line: str):
        """Inspect an MCU printf line and update internal state."""
        if "Processor Enabled" in line:
            self._state = self.STATE_STARTUP
        elif "Memory Loaded" in line:
            self._state = self.STATE_IDLE
        elif "Disabling Processor" in line:
            self._state = self.STATE_DISABLED
        elif "Entering Config Mode" in line:
            self._state = self.STATE_CONFIG
        elif "Config complete" in line:
            self._state = self.STATE_DISABLED
        elif "Data Complete" in line:
            self._state = self.STATE_READ_OUTPUT

    def _exit_binary_mode(self):
        self.ser.write(bytes([SLIP_END]))

    @property
    def state(self) -> str:
        """Current tracked state of the MCU state machine."""
        return self._state

    # ------------------------------------------------------------------
    # SLIP helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _slip_encode(data: bytes) -> bytes:
        out = bytearray()
        for b in data:
            if b == SLIP_END:
                out.extend([SLIP_ESC, SLIP_ESC_END])
            elif b == SLIP_ESC:
                out.extend([SLIP_ESC, SLIP_ESC_ESC])
            else:
                out.append(b)
        return bytes(out)

    def _build_slip_frame(
        self,
        write_size: int,
        write_addr: int,
        read_size: int,
        read_addr: int,
        payload: bytes,
    ) -> bytes:
        header = struct.pack("<IIII", write_size, write_addr, read_size, read_addr)
        return self._slip_encode(header + payload) + bytes([SLIP_END])

    # ------------------------------------------------------------------
    # Public commands
    # ------------------------------------------------------------------
    def disable(self):
        """Send 'disable' and wait for the MCU to enter STATE_DISABLED."""
        self.mode()
        if self._state == self.STATE_DISABLED:
            return
        self._send_text_cmd("disable")
        self._wait_for("Disabling Processor", timeout=3.0)

    def enable(self):
        """Send 'enable' and wait for startup (STATE_IDLE)."""
        self.mode()
        if self._state == self.STATE_IDLE:
            return
        self._send_text_cmd("enable")
        self._wait_for("Memory Loaded", timeout=10.0)

    def reset(self):
        """Trigger a hard reset of the Astra processor."""
        self._send_text_cmd("reset")
        self._wait_for("Astra Processor Reset", timeout=2.0)

    def mode(self):
        """Get current running mode or processor"""
        self._send_text_cmd("mode")
        try:
            mode = int(self._wait_for("RM:", timeout=2.0)[3])
        except:
            print("Attempting to exit binary mode")
            self._exit_binary_mode()
            self._send_text_cmd("mode")
            mode = int(self._wait_for("RM:", timeout=2.0)[3])
        self._state = self.stateDict[mode]


        

    # ------------------------------------------------------------------
    # Flash config uploads
    # ------------------------------------------------------------------
    def write_paramcfg(self, filepath_or_bytes, offset: int = 0, timeout: float = 90.0):                                                                                                              
        data = self._load_binary(filepath_or_bytes)                                                                                                                                                   
        PAGE_SIZE = 2048                                                                                                                                                                              

        self.mode()                                                                                                                                                                                     
        if self._state != self.STATE_DISABLED:                                                                                                                                                        
            self.disable()                                                                                                                                                                            
                                                                                                                                                                                                        
        self._send_text_cmd("pconfig")                                                                                                                                                                   
        self._wait_for("Entering Config Mode", timeout=2.0)                                                                                                                                           
                                                                                                                                                                                                        
        # 1. Send header ONLY — no trailing 0xC0                                                                                                                                                      
        header = struct.pack("<IIII", len(data), offset, 0, 0)                                                                                                                                        
        self.ser.write(self._slip_encode(header))                                                                                                                                                     
        self.ser.flush()                                                                                                                                                                              
                                                                                                                                                                                                        
        # 2. Wait for MCU to parse header                                                                                                                                                             
        self._wait_for("RDY", timeout=5.0)                                                                                                                                                            
                                                                                                                                                                                                        
        # 3. Stream payload in page-size chunks, wait for ACK after each                                                                                                                              
        for i in range(0, len(data), PAGE_SIZE):                                                                                                                                                      
            chunk = data[i:i + PAGE_SIZE]                                                                                                                                                             
            encoded = self._slip_encode(chunk)                                                                                                                                                        
                                                                                                                                                                                                        
            if i + PAGE_SIZE >= len(data):    
                print("END OF TRANSFER")                                                                                                                                                        
                # LAST chunk: append SLIP END to signal end of stream                                                                                                                                 
                encoded += bytes([SLIP_END])                                                                                                                                                          
                                                                                                                                                                                                        
            self.ser.write(encoded)                                                                                                                                                                   
            self.ser.flush()                                                                                                                                                                          
                                                                                                                                                                                                        
            if i + PAGE_SIZE < len(data):                                                                                                                                                             
                # Not the last page: wait for MCU erase+program before continuing                                                                                                                     
                self._wait_for("ACK", timeout=10.0)                                                                                                                                                   
                                                                                                                                                                                                        
        # 4. Wait for final cleanup                                                                                                                                                                   
        self._wait_for("Config complete", timeout=timeout)                                                                                                                                            
        return len(data)

    def write_instcfg(self, filepath_or_bytes, offset: int = 0, timeout: float = 90.0):                                                                                                               
        data = self._load_binary(filepath_or_bytes)                                                                                                                                                   
        PAGE_SIZE = 2048                                                                                                                                                                              

        self.mode()                                                                                                                                                                                    
        if self._state != self.STATE_DISABLED:                                                                                                                                                        
            self.disable()                                                                                                                                                                            
                                                                                                                                                                                                        
        self._send_text_cmd("iconfig")                                                                                                                                                                   
        self._wait_for("Entering Config Mode", timeout=2.0)                                                                                                                                           
                                                                                                                                                                                                        
        header = struct.pack("<IIII", len(data), offset, 0, 0)                                                                                                                                        
        self.ser.write(self._slip_encode(header))                                                                                                                                                     
        self.ser.flush()                                                                                                                                                                              
                                                                                                                                                                                                        
        self._wait_for("RDY", timeout=5.0)                                                                                                                                                            
                                                                                                                                                                                                        
        for i in range(0, len(data), PAGE_SIZE):                                                                                                                                                      
            chunk = data[i:i + PAGE_SIZE]                                                                                                                                                             
            encoded = self._slip_encode(chunk)                                                                                                                                                        
                                                                                                                                                                                                        
            if i + PAGE_SIZE >= len(data):                                                                                                                                                            
                encoded += bytes([SLIP_END])                                                                                                                                                          
                                                                                                                                                                                                        
            self.ser.write(encoded)                                                                                                                                                                   
            self.ser.flush()                                                                                                                                                                          
                                                                                                                                                                                                        
            if i + PAGE_SIZE < len(data):                                                                                                                                                             
                self._wait_for("ACK", timeout=10.0)                                                                                                                                                   
                                                                                                                                                                                                        
        self._wait_for("Config complete", timeout=timeout)                                                                                                                                            
        return len(data)

    @staticmethod
    def _load_binary(filepath_or_bytes) -> bytes:
        if isinstance(filepath_or_bytes, str):
            with open(filepath_or_bytes, "rb") as f:
                return f.read()
        return bytes(filepath_or_bytes)

    # ------------------------------------------------------------------
    # Activation run
    # ------------------------------------------------------------------
    def run_activation(
        self,
        data: list[int],
        read_size: int = 10,
        read_addr: int = 232,
        write_addr: int = 0,
        timeout: float = 10.0,
    ) -> list[int]:
        """
        Send activation data (list of 0..255 integers) to the Astra chip,
        wait for computation to finish, and return the output bytes.

        Parameters
        ----------
        data : list[int]
            Raw payload bytes to SLIP-encode and send.
        read_size : int
            How many result bytes to read back (default 60).
        read_addr : int
            SPI readback address for astra_transfer (default 0).
        write_addr : int
            SPI write address for astra_transfer (default 0).
        timeout : float
            Max seconds to wait for results.

        Returns
        -------
        list[int]
            Decoded output bytes from STATE_READ_OUTPUT.
        """
        t0 = time.perf_counter()

        
        if self._state != self.STATE_IDLE:
            self.mode()
            if self._state != self.STATE_IDLE:
                self.enable()
        t_mode = time.perf_counter()

        payload = bytes(data)
        frame = self._build_slip_frame(
            len(payload), write_addr, read_size, read_addr, payload
        )
        t_build = time.perf_counter()

        # 1. Tell MCU to enter activation mode
        self._send_text_cmd("run")
        # Small delay so the MCU can transition into STATE_SEND_ACTIVATION
        # before the first SLIP bytes hit the ISR.
        self._wait_for("READY", timeout=0.1)
        #time.sleep(0.05)
        self._state = self.STATE_SEND_ACTIVATION
        t_ready = time.perf_counter()

        # 2. Stream the SLIP frame
        
        self.ser.write(frame)
        self.ser.flush()
        t_send = time.perf_counter()

        # 3. Wait for "Data Complete" (end of binary mode -> STATE_READ_OUTPUT)
        # self._wait_for("Data Complete", timeout=timeout)
        self._state = self.STATE_READ_OUTPUT

        # 4. Collect "Activation Output:NN" lines
        outputs = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            for line in self._readlines(timeout=0.01):
                self._update_state_from_line(line)
                if "AO:" in line:
                    #print(line)
                    try:
                        
                        val = int(line.split(":", 1)[1].strip())
                        outputs.append(val)
                    except (ValueError, IndexError):
                        pass
                elif "done" in line:
                    # All outputs have been printed by this point

                    break

            if len(outputs) >= read_size:
                break
        t_collect = time.perf_counter()

        # MCU auto-transitions back to STATE_IDLE after STATE_READ_OUTPUT
        self._state = self.STATE_IDLE

        # print(
        #     f"[PROFILE] mode/enable: {(t_mode - t0)*1000:.3f} ms | "
        #     f"build_frame: {(t_build - t_mode)*1000:.3f} ms | "
        #     f"wait_ready: {(t_ready - t_build)*1000:.3f} ms | "
        #     f"send_frame: {(t_send - t_ready)*1000:.3f} ms | "
        #     f"collect_output: {(t_collect - t_send)*1000:.3f} ms | "
        #     f"total: {(t_collect - t0)*1000:.3f} ms"
        # )

        return outputs
    

if __name__ == "__main__":
    import glob
    import time
    import sys
    from natsort import natsorted

    inFiles = natsorted(glob.glob("Binaries\\Primary_Inputs\\*_input_bin.txt"))
    outFiles = natsorted(glob.glob("Binaries\\Output_Files\\true_op_layer13_sample*.txt"))
    if not inFiles or not outFiles:
        print("Missing required files check input and output directory.")
        sys.exit(1)

    matchCount = 0
    mismatchCount = 0
    mismatchList = []
        
    with AstraInterface("COM3", 115200) as astra:                                                                                                                                                       
        #Already enabled and idle                                                                                                                                                                       
                                                                                                                                                                    
        #astra.write_instcfg("Binaries\\Instructions\\instr_mem.bin")   
        #astra.write_paramcfg("Binaries\\Parameters\\param_mem.bin")
        i = 0
        #print("HELLO")
        for inFile, outFile in zip(inFiles, outFiles):
            i += 1
            with open(inFile) as f:
                inData = [int(n.replace("\n", ""),2) for n in f.readlines()]


            with open(outFile) as f:
                outData = [int(n.replace("\n", ""),16) for n in f.readlines()]

            #print([hex(n)[2:] for n in inData])
            startTime = time.monotonic_ns()
            outTest = astra.run_activation(inData, read_size=17, read_addr=256)[1:]
            endTime = time.monotonic_ns()

            print(inFile)
            print(outTest)
            print(outFile)
            print(outData)
            if outTest == outData:
                matchCount += 1
                print("MATCHING")
            else:
                mismatchList.append((inFile, outTest, outFile, outData))
                mismatchCount += 1
                print("NOT MATCHING")
            print(f"Compute Time: {(endTime-startTime)/1000000}ms")
            print("\n")
            # if i >= 300:
            #     break

    print(f"{matchCount}/{len(inFiles)} Are Matching: {(matchCount/len(inFiles))*100}% Accuracy")

    print(f"Mismatches: {mismatchList}")
    

