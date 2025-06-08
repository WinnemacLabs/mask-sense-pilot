# Pressure-Based Mask Fit Testing Protocol

## Required Tools

1. 3-Axis Differential Pressure Logger (Custom PCB)
2. WRPAS Dual Channel Particle Counter (TSI)
3. N95 Respirator, fitted with three 1/16" hose barbs and one TSI probe

## Setup

### Masks

Prepare two different N95 masks using the following procedure:

1. Use 5mm Circle punch to create holes in mask at the center, 20 mm to the left of center (X-axis), and 20mm above (Y-axis). Left is from the wearer's persepective.
2. Secure three [1/16" hose barb bulkhead fittings](https://www.amazon.com/Polypropylene-Connector-Heat-Resistant-Chemically-Processing/dp/B0F21F4XVT?sr=8-5) to these three holes. Snip the luer lock portion off so there is no contact with the user's face.
3. Use TSI sampling probe kit to insert a probe 20mm to the right of center. 

### WRPAS

1. Perform WRPAS wick-wetting procedure as described in the instruction manual
2. Connect sampling tubes to WRPAS - clear tube at top (CH1) and blue tube at bottom (CH2)
3. Connect clear tube to TSI sampling port on N95 mask
4. Turn on WRPAS, make sure system check passes
5. Connect WRPAS to computer with mini USB cable


### 3-Axis Pressure Logger

1. Connect REF tube to center hose barb on mask.
2. Connect Y tube to upper hose barb.
3. Connext X tube to left hose barb.
4. Connect pressure logger to computer with micro usb cable. 

### Computer

1. Start pressure-particles-ingestion.py (e.g. `python pressure-particles-ingestion.py)
2. Select the ports corresponding to the pressure logger and then the WRPAS (e.g. `COM4` (Windows) or `/dev/tty*` (Mac)) 

## Experimental Procedure

1. Allow the participant to fit mask onto their face
2. Ask the user to hold their breath, type `zero` command to zero out pressure sensors
3. type `start P<#>_<mask-type>_quiet_breathing` to begin quiet breathing recording
4. Participant performs three minutes of quiet breathing while sitting down
5. `stop` to stop recording. 
6. `start P<#>_<mask-type>_deep_breathing`
7. Participant performs 1 minute of deep breathing (don't hyperventilate)
8. `stop` to stop recording.
9. `start P<#>_<mask-type>_rainbow`
10. Read aloud the rainbow passage: 

>>> When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it.When a man looks for something beyond reach, his friends say he is looking for the pot of gold at the end of the rainbow. 

11. Introduce face-seal leak using polyfilla on right side of mask. Repeat above procedure

12. Repeat above procedure with mask #2

