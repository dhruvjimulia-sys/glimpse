#include "powerandarea.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "isa.h"
#include "utils/program_utils.h"

#define MILLI_ORDER_OF_MAGNITUDE 1e3
#define NANO_ORDER_OF_MAGNITUDE 1e9
#define MILLI_TO_MICRO_ORDER_OF_MAGNITUDE 1e3
#define MICRO_TO_NANO_ORDER_OF_MAGNITUDE 1e3

// From latest SRVC paper
// ASSUME neighbour-to-neighbour communication bottleneck + more assumptions
// ASSUME no interconnect area and power
// ASSUME no ADC and photodetector area and power
#define CLOCK_FREQUENCY 2e8  // 200 MHz
// ASSUME neighbour-to-neighbour communication
#define TARGET_TECHNOLOGY 65  // 65nm // Must be <= 90nm due to CACTI
// ASSUME supply voltage of 1.0V works
#define SUPPLY_VOLTAGE 1.0    // 1V
#define TEMPERATURE \
    300  // in K // must be a multiple of 10 // and must be between 300 and 400
         // inclusive

// From Skywater 130nm PDK
#define SOURCE_DATA_TECHNOLOGY 130  // 130nm
// TODO Copy correct values here
#define AREA_OF_FULL_ADDER_130_NM 20.0192    // in um^2
#define AREA_OF_MULTIPLEXER_130_NM 11.2608  // in um^2

#define NUMBER_TECH_FLAVORS 4

#define MEMORY_MULTIPLIER 88
#define REGISTER_MULTIPLIER 128

enum ram_cell_tech_type_num {
    itrs_hp = 0,
    itrs_lstp = 1,
    itrs_lop = 2,
    lp_dram = 3,
    comm_dram = 4
};
// for lp_dram and comm_dram, n_to_p_eff_curr_drv_ratio, I_off_n, I_g_on_n not
// defined (only I_off_n defined)

#define TECH_TYPE itrs_lop

double scaleAreaBasedOnTechnology(double area, size_t old_tech,
    size_t new_tech);
double scalePowerBasedOnTechnology(double power, size_t old_tech,
     size_t new_tech);
TechnologyParameter getTechnologyParams(int technology);
double pmos_to_nmos_sz_ratio(TechnologyParameter g_tp);
double getLogicScalingFactor(size_t source_technology, size_t target_technology);
double cmos_Ig_leakage(double nWidth, double pWidth, TechnologyParameter g_tp);
double cmos_Isub_leakage(double nWidth, double pWidth,
    TechnologyParameter g_tp);
CACTIResult getCACTIResult(std::string filename, size_t vliwWidth);

// Compute area and power functions
// in um^2
double getComputeArea(size_t vliwWidth) {
    double oneComputeUnitArea = scaleAreaBasedOnTechnology(
        AREA_OF_FULL_ADDER_130_NM * 4 + AREA_OF_MULTIPLEXER_130_NM,
        SOURCE_DATA_TECHNOLOGY, TARGET_TECHNOLOGY);  // in um^2
    double computeArea =
        oneComputeUnitArea * vliwWidth;
    return computeArea;  // in um^2
}

// in W
double getComputeSubthresholdLeakage(size_t vliwWidth) {
    TechnologyParameter g_tp = getTechnologyParams(TARGET_TECHNOLOGY);
    // area must be in um^2
    return getComputeArea(vliwWidth) * g_tp.scaling_factor.core_tx_density *
           cmos_Isub_leakage(
               g_tp.min_w_nmos_,
               g_tp.min_w_nmos_ * pmos_to_nmos_sz_ratio(g_tp),
               g_tp) *
           SUPPLY_VOLTAGE / (2 * 1e7);  // unit W
}

// in W
double getComputeGateLeakage(size_t vliwWidth) {
    TechnologyParameter g_tp = getTechnologyParams(TARGET_TECHNOLOGY);
    // area must be in um^2
    return getComputeArea(vliwWidth) * g_tp.scaling_factor.core_tx_density *
           cmos_Ig_leakage(g_tp.min_w_nmos_,
                           g_tp.min_w_nmos_ * pmos_to_nmos_sz_ratio(g_tp),
                           g_tp) *
           SUPPLY_VOLTAGE / (2 * 1e7);  // unit W
}

// in W
double getComputeDynamicPower(Program program) {
    double per_access_energy =
        (1.15 / 3 / 1e9 / 4 / 1.3 / 1.3 * SUPPLY_VOLTAGE * SUPPLY_VOLTAGE *
         (TARGET_TECHNOLOGY / 90.0)) /
        64;  // This is per cycle energy(nJ)
    // PIPELINING: CLOCK_FREQUENCY needs to be divided in four to get effective clock frequency for ALU (in non-pipelining case)
    return (numComputeAccesses(program) * per_access_energy * (CLOCK_FREQUENCY / 4)) /
           NANO_ORDER_OF_MAGNITUDE;  // in W
}

// in um^2
double getMemoryArea(size_t vliwWidth) {
    CACTIResult memoryResult = getCACTIResult("memory.cfg", vliwWidth);
    CACTIResult registersResult = getCACTIResult("registers.cfg", vliwWidth);
    // area from CACTI is in mm^2
    return (((memoryResult.height * memoryResult.width) / MEMORY_MULTIPLIER +
           ((registersResult.height * registersResult.width) / REGISTER_MULTIPLIER) * vliwWidth) * MILLI_TO_MICRO_ORDER_OF_MAGNITUDE) * MILLI_TO_MICRO_ORDER_OF_MAGNITUDE;
}

// in W
double getMemorySubthresholdLeakage(size_t vliwWidth) {
    CACTIResult memoryResult = getCACTIResult("memory.cfg", vliwWidth);
    CACTIResult registersResult = getCACTIResult("registers.cfg", vliwWidth);
    // leakage_power from CACTI is in mW
    return (memoryResult.leakage_power / MEMORY_MULTIPLIER +
           (registersResult.leakage_power / REGISTER_MULTIPLIER) * vliwWidth) / MILLI_ORDER_OF_MAGNITUDE;
}

// in W
double getMemoryGateLeakage(size_t vliwWidth) {
    CACTIResult memoryResult = getCACTIResult("memory.cfg", vliwWidth);
    CACTIResult registersResult = getCACTIResult("registers.cfg", vliwWidth);
    // gate_leakage_power from CACTI is in mW
    return (memoryResult.gate_leakage_power / MEMORY_MULTIPLIER +
           (registersResult.gate_leakage_power / REGISTER_MULTIPLIER) * vliwWidth) / MILLI_ORDER_OF_MAGNITUDE;
}

// in W
double getMemoryDynamicPower(Program program) {
    CACTIResult memoryResult = getCACTIResult("memory.cfg", program.vliwWidth);
    CACTIResult registersResult =
        getCACTIResult("registers.cfg", program.vliwWidth);
    // dynamic_read_energy_per_access from CACTI is in nJ
    // PIPELINING: CLOCK_FREQUENCY needs to be divided in four to get effective clock frequency for various components
    // This is equivalent to instruction cycle clock frequency
    return ((((memoryResult.dynamic_read_energy_per_access *
           numMemoryReadAccesses(program) +
           memoryResult.dynamic_write_energy_per_access *
               numMemoryWriteAccesses(program)) / MEMORY_MULTIPLIER +
           ((registersResult.dynamic_read_energy_per_access *
               numRegisterReadAccesses(program) +
           registersResult.dynamic_write_energy_per_access *
               numRegisterWriteAccesses(program)) / REGISTER_MULTIPLIER)) / program.instructionCount) * (CLOCK_FREQUENCY / 4)) / NANO_ORDER_OF_MAGNITUDE;
}

// Technology scaling parameters below taken from McPAT
// scaling_factor.logic_scaling_co_eff
// scaling_factor.core_tx_density
// min_w_nmos_
// peri_global.n_to_p_eff_curr_drv_ratio
// peri_global.I_off_n
// peri_global.I_off_p
// peri_global.I_g_on_n
// peri_global.I_g_on_p
TechnologyParameter getTechnologyParams(int technology) {
    double curr_logic_scaling_co_eff;
    size_t iter, tech, tech_lo, tech_hi;
    double curr_core_tx_density =
        0;  // this is density per um^2; 90, ...22nm based on Intel Penryn
    double curr_alpha;
    double vdd[NUMBER_TECH_FLAVORS];  // default vdd from itrs
    double n_to_p_eff_curr_drv_ratio[NUMBER_TECH_FLAVORS];
    double I_off_n[NUMBER_TECH_FLAVORS][101];
    double I_g_on_n[NUMBER_TECH_FLAVORS][101];
    TechnologyParameter g_tp;

    if (technology < 181 && technology > 179) {
        tech_lo = 180;
        tech_hi = 180;
    } else if (technology < 91 && technology > 89) {
        tech_lo = 90;
        tech_hi = 90;
    } else if (technology < 66 && technology > 64) {
        tech_lo = 65;
        tech_hi = 65;
    } else if (technology < 46 && technology > 44) {
        tech_lo = 45;
        tech_hi = 45;
    } else if (technology < 33 && technology > 31) {
        tech_lo = 32;
        tech_hi = 32;
    } else if (technology < 23 && technology > 21) {
        tech_lo = 22;
        tech_hi = 22;
    } else if (technology < 180 && technology > 90) {
        tech_lo = 180;
        tech_hi = 90;
    } else if (technology < 90 && technology > 65) {
        tech_lo = 90;
        tech_hi = 65;
    } else if (technology < 65 && technology > 45) {
        tech_lo = 65;
        tech_hi = 45;
    } else if (technology < 45 && technology > 32) {
        tech_lo = 45;
        tech_hi = 32;
    } else if (technology < 32 && technology > 22) {
        tech_lo = 32;
        tech_hi = 22;
    } else {
        std::cout << "Invalid technology nodes" << std::endl;
        exit(0);
    }

    for (iter = 0; iter <= 1; ++iter) {
        // linear interpolation
        if (iter == 0) {
            tech = tech_lo;
            if (tech_lo == tech_hi) {
                curr_alpha = 1;
            } else {
                curr_alpha = (((double) technology) - tech_hi) / (((double) tech_lo) - tech_hi);
            }
        } else {
            tech = tech_hi;
            if (tech_lo == tech_hi) {
                break;
            } else {
                curr_alpha = (tech_lo - ((double) technology)) / (((double) tech_lo) - tech_hi);
            }
        }

        if (tech == 180) {
            // 180nm technology-node. Corresponds to year 1999 in ITRS
            // Only HP transistor was of interest that 180nm since leakage power
            // was not a big issue. Performance was the king MASTAR does not
            // contain data for 0.18um process. The following parameters are
            // projected based on ITRS 2000 update and IBM 0.18 Cu Spice input
            // 180nm does not support DVS Empirical undifferetiated core/FU
            // coefficient
            curr_logic_scaling_co_eff = 1.5;  // linear scaling from 90nm
            curr_core_tx_density = 1.25 * 0.7 * 0.7 * 0.4;
            n_to_p_eff_curr_drv_ratio[0] = 2.45;

            I_off_n[0][0] = 7e-10;  // A/micron
            I_off_n[0][10] = 8.26e-10;
            I_off_n[0][20] = 9.74e-10;
            I_off_n[0][30] = 1.15e-9;
            I_off_n[0][40] = 1.35e-9;
            I_off_n[0][50] = 1.60e-9;
            I_off_n[0][60] = 1.88e-9;
            I_off_n[0][70] = 2.29e-9;
            I_off_n[0][80] = 2.70e-9;
            I_off_n[0][90] = 3.19e-9;
            I_off_n[0][100] = 3.76e-9;

            I_g_on_n[0][0] = 1.65e-10;  // A/micron
            I_g_on_n[0][10] = 1.65e-10;
            I_g_on_n[0][20] = 1.65e-10;
            I_g_on_n[0][30] = 1.65e-10;
            I_g_on_n[0][40] = 1.65e-10;
            I_g_on_n[0][50] = 1.65e-10;
            I_g_on_n[0][60] = 1.65e-10;
            I_g_on_n[0][70] = 1.65e-10;
            I_g_on_n[0][80] = 1.65e-10;
            I_g_on_n[0][90] = 1.65e-10;
            I_g_on_n[0][100] = 1.65e-10;
        }

        if (tech == 90) {
            // Empirical undifferetiated core/FU coefficient
            curr_logic_scaling_co_eff = 1;
            curr_core_tx_density = 1.25 * 0.7 * 0.7;

            n_to_p_eff_curr_drv_ratio[0] = 2.45;

            vdd[0] = 1.2;
            n_to_p_eff_curr_drv_ratio[0] = 2.45;
            I_off_n[0][0] =
                3.24e-8 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);  // A/micron
            I_off_n[0][10] = 4.01e-8 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][20] = 4.90e-8 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][30] = 5.92e-8 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][40] = 7.08e-8 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][50] = 8.38e-8 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][60] = 9.82e-8 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][70] = 1.14e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][80] = 1.29e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][90] = 1.43e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][100] = 1.54e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);

            I_g_on_n[0][0] = 1.65e-8;  // A/micron
            I_g_on_n[0][10] = 1.65e-8;
            I_g_on_n[0][20] = 1.65e-8;
            I_g_on_n[0][30] = 1.65e-8;
            I_g_on_n[0][40] = 1.65e-8;
            I_g_on_n[0][50] = 1.65e-8;
            I_g_on_n[0][60] = 1.65e-8;
            I_g_on_n[0][70] = 1.65e-8;
            I_g_on_n[0][80] = 1.65e-8;
            I_g_on_n[0][90] = 1.65e-8;
            I_g_on_n[0][100] = 1.65e-8;

            vdd[1] = 1.3;
            n_to_p_eff_curr_drv_ratio[1] = 2.44;
            I_off_n[1][0] = 2.81e-12 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][10] = 4.76e-12 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][20] = 7.82e-12 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][30] = 1.25e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][40] = 1.94e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][50] = 2.94e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][60] = 4.36e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][70] = 6.32e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][80] = 8.95e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][90] = 1.25e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][100] = 1.7e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);

            I_g_on_n[1][0] = 3.87e-11;  // A/micron
            I_g_on_n[1][10] = 3.87e-11;
            I_g_on_n[1][20] = 3.87e-11;
            I_g_on_n[1][30] = 3.87e-11;
            I_g_on_n[1][40] = 3.87e-11;
            I_g_on_n[1][50] = 3.87e-11;
            I_g_on_n[1][60] = 3.87e-11;
            I_g_on_n[1][70] = 3.87e-11;
            I_g_on_n[1][80] = 3.87e-11;
            I_g_on_n[1][90] = 3.87e-11;
            I_g_on_n[1][100] = 3.87e-11;

            vdd[2] = 0.9;
            n_to_p_eff_curr_drv_ratio[2] = 2.54;
            I_off_n[2][0] = 2.14e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][10] = 2.9e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][20] = 3.87e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][30] = 5.07e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][40] = 6.54e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][50] = 8.27e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][60] = 1.02e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][70] = 1.20e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][80] = 1.36e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][90] = 1.52e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][100] = 1.73e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);

            I_g_on_n[2][0] = 4.31e-8;  // A/micron
            I_g_on_n[2][10] = 4.31e-8;
            I_g_on_n[2][20] = 4.31e-8;
            I_g_on_n[2][30] = 4.31e-8;
            I_g_on_n[2][40] = 4.31e-8;
            I_g_on_n[2][50] = 4.31e-8;
            I_g_on_n[2][60] = 4.31e-8;
            I_g_on_n[2][70] = 4.31e-8;
            I_g_on_n[2][80] = 4.31e-8;
            I_g_on_n[2][90] = 4.31e-8;
            I_g_on_n[2][100] = 4.31e-8;

            if (TECH_TYPE == lp_dram) {
                I_off_n[3][0] = 1.42e-11;
                I_off_n[3][10] = 2.25e-11;
                I_off_n[3][20] = 3.46e-11;
                I_off_n[3][30] = 5.18e-11;
                I_off_n[3][40] = 7.58e-11;
                I_off_n[3][50] = 1.08e-10;
                I_off_n[3][60] = 1.51e-10;
                I_off_n[3][70] = 2.02e-10;
                I_off_n[3][80] = 2.57e-10;
                I_off_n[3][90] = 3.14e-10;
                I_off_n[3][100] = 3.85e-10;
            } else if (TECH_TYPE == comm_dram) {
                I_off_n[3][0] = 5.80e-15;
                I_off_n[3][10] = 1.21e-14;
                I_off_n[3][20] = 2.42e-14;
                I_off_n[3][30] = 4.65e-14;
                I_off_n[3][40] = 8.60e-14;
                I_off_n[3][50] = 1.54e-13;
                I_off_n[3][60] = 2.66e-13;
                I_off_n[3][70] = 4.45e-13;
                I_off_n[3][80] = 7.17e-13;
                I_off_n[3][90] = 1.11e-12;
                I_off_n[3][100] = 1.67e-12;
            }
        }

        if (tech == 65) {
            // 65nm technology-node. Corresponds to year 2007 in ITRS
            // ITRS HP device type
            // Empirical undifferetiated core/FU coefficient
            curr_logic_scaling_co_eff =
                0.7;  // Rather than scale proportionally to square of feature
                      // size, only scale linearly according to IBM cell
                      // processor
            curr_core_tx_density = 1.25 * 0.7;

            vdd[0] = 1.1;
            n_to_p_eff_curr_drv_ratio[0] = 2.41;
            I_off_n[0][0] = 1.96e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][10] = 2.29e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][20] = 2.66e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][30] = 3.05e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][40] = 3.49e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][50] = 3.95e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][60] = 4.45e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][70] = 4.97e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][80] = 5.48e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][90] = 5.94e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][100] = 6.3e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_g_on_n[0][0] = 4.09e-8;  // A/micron
            I_g_on_n[0][10] = 4.09e-8;
            I_g_on_n[0][20] = 4.09e-8;
            I_g_on_n[0][30] = 4.09e-8;
            I_g_on_n[0][40] = 4.09e-8;
            I_g_on_n[0][50] = 4.09e-8;
            I_g_on_n[0][60] = 4.09e-8;
            I_g_on_n[0][70] = 4.09e-8;
            I_g_on_n[0][80] = 4.09e-8;
            I_g_on_n[0][90] = 4.09e-8;
            I_g_on_n[0][100] = 4.09e-8;

            vdd[1] = 1.2;
            n_to_p_eff_curr_drv_ratio[1] = 2.23;
            I_off_n[1][0] = 9.12e-12 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][10] = 1.49e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][20] = 2.36e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][30] = 3.64e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][40] = 5.48e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][50] = 8.05e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][60] = 1.15e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][70] = 1.59e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][80] = 2.1e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][90] = 2.62e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][100] = 3.21e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_g_on_n[1][0] = 1.09e-10;  // A/micron
            I_g_on_n[1][10] = 1.09e-10;
            I_g_on_n[1][20] = 1.09e-10;
            I_g_on_n[1][30] = 1.09e-10;
            I_g_on_n[1][40] = 1.09e-10;
            I_g_on_n[1][50] = 1.09e-10;
            I_g_on_n[1][60] = 1.09e-10;
            I_g_on_n[1][70] = 1.09e-10;
            I_g_on_n[1][80] = 1.09e-10;
            I_g_on_n[1][90] = 1.09e-10;
            I_g_on_n[1][100] = 1.09e-10;

            vdd[2] = 0.8;
            n_to_p_eff_curr_drv_ratio[2] = 2.28;
            I_off_n[2][0] = 4.9e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][10] = 6.49e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][20] = 8.45e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][30] = 1.08e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][40] = 1.37e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][50] = 1.71e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][60] = 2.09e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][70] = 2.48e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][80] = 2.84e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][90] = 3.13e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][100] = 3.42e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);

            I_g_on_n[2][0] = 9.61e-9;  // A/micron
            I_g_on_n[2][10] = 9.61e-9;
            I_g_on_n[2][20] = 9.61e-9;
            I_g_on_n[2][30] = 9.61e-9;
            I_g_on_n[2][40] = 9.61e-9;
            I_g_on_n[2][50] = 9.61e-9;
            I_g_on_n[2][60] = 9.61e-9;
            I_g_on_n[2][70] = 9.61e-9;
            I_g_on_n[2][80] = 9.61e-9;
            I_g_on_n[2][90] = 9.61e-9;
            I_g_on_n[2][100] = 9.61e-9;

            if (TECH_TYPE == lp_dram) {
                I_off_n[3][0] = 2.23e-11;
                I_off_n[3][10] = 3.46e-11;
                I_off_n[3][20] = 5.24e-11;
                I_off_n[3][30] = 7.75e-11;
                I_off_n[3][40] = 1.12e-10;
                I_off_n[3][50] = 1.58e-10;
                I_off_n[3][60] = 2.18e-10;
                I_off_n[3][70] = 2.88e-10;
                I_off_n[3][80] = 3.63e-10;
                I_off_n[3][90] = 4.41e-10;
                I_off_n[3][100] = 5.36e-10;
            } else if (TECH_TYPE == comm_dram) {
                I_off_n[3][0] = 1.80e-14;
                I_off_n[3][10] = 3.64e-14;
                I_off_n[3][20] = 7.03e-14;
                I_off_n[3][30] = 1.31e-13;
                I_off_n[3][40] = 2.35e-13;
                I_off_n[3][50] = 4.09e-13;
                I_off_n[3][60] = 6.89e-13;
                I_off_n[3][70] = 1.13e-12;
                I_off_n[3][80] = 1.78e-12;
                I_off_n[3][90] = 2.71e-12;
                I_off_n[3][100] = 3.99e-12;
            }
        }

        if (tech == 45) {
            // 45nm technology-node. Corresponds to year 2010 in ITRS
            // ITRS HP device type
            // Empirical undifferetiated core/FU coefficient
            curr_logic_scaling_co_eff = 0.7 * 0.7;
            curr_core_tx_density = 1.25;

            vdd[0] = 1.0;
            n_to_p_eff_curr_drv_ratio[0] = 2.41;
            I_off_n[0][0] = 2.8e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][10] = 3.28e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][20] = 3.81e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][30] = 4.39e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][40] = 5.02e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][50] = 5.69e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][60] = 6.42e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][70] = 7.2e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][80] = 8.03e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][90] = 8.91e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);
            I_off_n[0][100] = 9.84e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 4);

            I_g_on_n[0][0] = 3.59e-8;  // A/micron
            I_g_on_n[0][10] = 3.59e-8;
            I_g_on_n[0][20] = 3.59e-8;
            I_g_on_n[0][30] = 3.59e-8;
            I_g_on_n[0][40] = 3.59e-8;
            I_g_on_n[0][50] = 3.59e-8;
            I_g_on_n[0][60] = 3.59e-8;
            I_g_on_n[0][70] = 3.59e-8;
            I_g_on_n[0][80] = 3.59e-8;
            I_g_on_n[0][90] = 3.59e-8;
            I_g_on_n[0][100] = 3.59e-8;

            vdd[1] = 1.1;
            n_to_p_eff_curr_drv_ratio[1] = 2.23;
            I_off_n[1][0] = 1.01e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][10] = 1.65e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][20] = 2.62e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][30] = 4.06e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][40] = 6.12e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][50] = 9.02e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][60] = 1.3e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][70] = 1.83e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][80] = 2.51e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][90] = 3.29e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);
            I_off_n[1][100] = 4.1e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 4);

            I_g_on_n[1][0] = 9.47e-12;  // A/micron
            I_g_on_n[1][10] = 9.47e-12;
            I_g_on_n[1][20] = 9.47e-12;
            I_g_on_n[1][30] = 9.47e-12;
            I_g_on_n[1][40] = 9.47e-12;
            I_g_on_n[1][50] = 9.47e-12;
            I_g_on_n[1][60] = 9.47e-12;
            I_g_on_n[1][70] = 9.47e-12;
            I_g_on_n[1][80] = 9.47e-12;
            I_g_on_n[1][90] = 9.47e-12;
            I_g_on_n[1][100] = 9.47e-12;

            vdd[2] = 0.7;
            n_to_p_eff_curr_drv_ratio[2] = 2.28;
            I_off_n[2][0] = 4.03e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][10] = 5.02e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][20] = 6.18e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][30] = 7.51e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][40] = 9.04e-9 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][50] = 1.08e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][60] = 1.27e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][70] = 1.47e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][80] = 1.66e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][90] = 1.84e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][100] = 2.03e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);

            I_g_on_n[2][0] = 3.24e-8;  // A/micron
            I_g_on_n[2][10] = 4.01e-8;
            I_g_on_n[2][20] = 4.90e-8;
            I_g_on_n[2][30] = 5.92e-8;
            I_g_on_n[2][40] = 7.08e-8;
            I_g_on_n[2][50] = 8.38e-8;
            I_g_on_n[2][60] = 9.82e-8;
            I_g_on_n[2][70] = 1.14e-7;
            I_g_on_n[2][80] = 1.29e-7;
            I_g_on_n[2][90] = 1.43e-7;
            I_g_on_n[2][100] = 1.54e-7;

            if (TECH_TYPE == lp_dram) {
                I_off_n[3][0] = 2.54e-11;
                I_off_n[3][10] = 3.94e-11;
                I_off_n[3][20] = 5.95e-11;
                I_off_n[3][30] = 8.79e-11;
                I_off_n[3][40] = 1.27e-10;
                I_off_n[3][50] = 1.79e-10;
                I_off_n[3][60] = 2.47e-10;
                I_off_n[3][70] = 3.31e-10;
                I_off_n[3][80] = 4.26e-10;
                I_off_n[3][90] = 5.27e-10;
                I_off_n[3][100] = 6.46e-10;
            } else if (TECH_TYPE == comm_dram) {
                I_off_n[3][0] = 1.31e-14;
                I_off_n[3][10] = 2.68e-14;
                I_off_n[3][20] = 5.25e-14;
                I_off_n[3][30] = 9.88e-14;
                I_off_n[3][40] = 1.79e-13;
                I_off_n[3][50] = 3.15e-13;
                I_off_n[3][60] = 5.36e-13;
                I_off_n[3][70] = 8.86e-13;
                I_off_n[3][80] = 1.42e-12;
                I_off_n[3][90] = 2.20e-12;
                I_off_n[3][100] = 3.29e-12;
            }
        }

        if (tech == 32) {
            // Empirical undifferetiated core/FU coefficient
            curr_logic_scaling_co_eff = 0.7 * 0.7 * 0.7;
            curr_core_tx_density = 1.25 / 0.7;

            vdd[0] = 0.9;
            n_to_p_eff_curr_drv_ratio[0] = 2.41;
            I_off_n[0][0] = 1.52e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][10] = 1.55e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][20] = 1.59e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][30] = 1.68e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][40] = 1.90e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][50] = 2.69e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][60] = 5.32e-7 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][70] = 1.02e-6 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][80] = 1.62e-6 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][90] = 2.73e-6 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][100] = 6.1e-6 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);

            I_g_on_n[0][0] = 6.55e-8;  // A/micron
            I_g_on_n[0][10] = 6.55e-8;
            I_g_on_n[0][20] = 6.55e-8;
            I_g_on_n[0][30] = 6.55e-8;
            I_g_on_n[0][40] = 6.55e-8;
            I_g_on_n[0][50] = 6.55e-8;
            I_g_on_n[0][60] = 6.55e-8;
            I_g_on_n[0][70] = 6.55e-8;
            I_g_on_n[0][80] = 6.55e-8;
            I_g_on_n[0][90] = 6.55e-8;
            I_g_on_n[0][100] = 6.55e-8;

            vdd[1] = 1;
            n_to_p_eff_curr_drv_ratio[1] = 2.23;
            I_off_n[1][0] = 2.06e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][10] = 3.30e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][20] = 5.15e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][30] = 7.83e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][40] = 1.16e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][50] = 1.69e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][60] = 2.40e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][70] = 3.34e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][80] = 4.54e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][90] = 5.96e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][100] = 7.44e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);

            I_g_on_n[1][0] = 3.73e-11;  // A/micron
            I_g_on_n[1][10] = 3.73e-11;
            I_g_on_n[1][20] = 3.73e-11;
            I_g_on_n[1][30] = 3.73e-11;
            I_g_on_n[1][40] = 3.73e-11;
            I_g_on_n[1][50] = 3.73e-11;
            I_g_on_n[1][60] = 3.73e-11;
            I_g_on_n[1][70] = 3.73e-11;
            I_g_on_n[1][80] = 3.73e-11;
            I_g_on_n[1][90] = 3.73e-11;
            I_g_on_n[1][100] = 3.73e-11;

            vdd[2] = 0.6;
            n_to_p_eff_curr_drv_ratio[2] = 2.28;
            I_off_n[2][0] = 5.94e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][10] = 7.23e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][20] = 8.7e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][30] = 1.04e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][40] = 1.22e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][50] = 1.43e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][60] = 1.65e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][70] = 1.90e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][80] = 2.15e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][90] = 2.39e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][100] = 2.63e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);

            I_g_on_n[2][0] = 2.93e-9;  // A/micron
            I_g_on_n[2][10] = 2.93e-9;
            I_g_on_n[2][20] = 2.93e-9;
            I_g_on_n[2][30] = 2.93e-9;
            I_g_on_n[2][40] = 2.93e-9;
            I_g_on_n[2][50] = 2.93e-9;
            I_g_on_n[2][60] = 2.93e-9;
            I_g_on_n[2][70] = 2.93e-9;
            I_g_on_n[2][80] = 2.93e-9;
            I_g_on_n[2][90] = 2.93e-9;
            I_g_on_n[2][100] = 2.93e-9;

            if (TECH_TYPE == lp_dram) {
                I_off_n[3][0] = 3.57e-11;
                I_off_n[3][10] = 5.51e-11;
                I_off_n[3][20] = 8.27e-11;
                I_off_n[3][30] = 1.21e-10;
                I_off_n[3][40] = 1.74e-10;
                I_off_n[3][50] = 2.45e-10;
                I_off_n[3][60] = 3.38e-10;
                I_off_n[3][70] = 4.53e-10;
                I_off_n[3][80] = 5.87e-10;
                I_off_n[3][90] = 7.29e-10;
                I_off_n[3][100] = 8.87e-10;
            } else if (TECH_TYPE == comm_dram) {
                I_off_n[3][0] = 3.63e-14;
                I_off_n[3][10] = 7.18e-14;
                I_off_n[3][20] = 1.36e-13;
                I_off_n[3][30] = 2.49e-13;
                I_off_n[3][40] = 4.41e-13;
                I_off_n[3][50] = 7.55e-13;
                I_off_n[3][60] = 1.26e-12;
                I_off_n[3][70] = 2.03e-12;
                I_off_n[3][80] = 3.19e-12;
                I_off_n[3][90] = 4.87e-12;
                I_off_n[3][100] = 7.16e-12;
            }
        }

        if (tech == 22) {
            curr_logic_scaling_co_eff = 0.7 * 0.7 * 0.7 * 0.7;
            curr_core_tx_density = 1.25 / 0.7 / 0.7;

            vdd[0] = 0.8;
            n_to_p_eff_curr_drv_ratio[0] =
                2;  // Wpmos/Wnmos = 2 in 2007 MASTAR. Look i
            I_off_n[0][0] = 1.52e-7 / 1.5 * 1.2 *
                            pow(SUPPLY_VOLTAGE / (vdd[0]),
                                2);  // From 22nm, leakage current are directly
                                     // from ITRS report rather than MASTAR,
                                     // since MASTAR has serious bugs there.
            I_off_n[0][10] =
                1.55e-7 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][20] =
                1.59e-7 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][30] =
                1.68e-7 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][40] =
                1.90e-7 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][50] =
                2.69e-7 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][60] =
                5.32e-7 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][70] =
                1.02e-6 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][80] =
                1.62e-6 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][90] =
                2.73e-6 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            I_off_n[0][100] =
                6.1e-6 / 1.5 * 1.2 * pow(SUPPLY_VOLTAGE / (vdd[0]), 2);
            // for 22nm DG HP
            I_g_on_n[0][0] = 1.81e-9;  // A/micron
            I_g_on_n[0][10] = 1.81e-9;
            I_g_on_n[0][20] = 1.81e-9;
            I_g_on_n[0][30] = 1.81e-9;
            I_g_on_n[0][40] = 1.81e-9;
            I_g_on_n[0][50] = 1.81e-9;
            I_g_on_n[0][60] = 1.81e-9;
            I_g_on_n[0][70] = 1.81e-9;
            I_g_on_n[0][80] = 1.81e-9;
            I_g_on_n[0][90] = 1.81e-9;
            I_g_on_n[0][100] = 1.81e-9;

            vdd[1] = 0.8;
            n_to_p_eff_curr_drv_ratio[1] = 2;
            I_off_n[1][0] = 2.43e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][10] = 4.85e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][20] = 9.68e-11 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][30] = 1.94e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][40] = 3.87e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][50] = 7.73e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][60] = 3.55e-10 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][70] = 3.09e-9 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][80] = 6.19e-9 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][90] = 1.24e-8 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);
            I_off_n[1][100] = 2.48e-8 * pow(SUPPLY_VOLTAGE / (vdd[1]), 1);

            I_g_on_n[1][0] = 4.51e-10;  // A/micron
            I_g_on_n[1][10] = 4.51e-10;
            I_g_on_n[1][20] = 4.51e-10;
            I_g_on_n[1][30] = 4.51e-10;
            I_g_on_n[1][40] = 4.51e-10;
            I_g_on_n[1][50] = 4.51e-10;
            I_g_on_n[1][60] = 4.51e-10;
            I_g_on_n[1][70] = 4.51e-10;
            I_g_on_n[1][80] = 4.51e-10;
            I_g_on_n[1][90] = 4.51e-10;
            I_g_on_n[1][100] = 4.51e-10;

            vdd[2] = 0.6;
            n_to_p_eff_curr_drv_ratio[2] = 2;
            I_off_n[2][0] = 1.31e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][10] = 2.60e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][20] = 5.14e-8 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][30] = 1.02e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][40] = 2.02e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][50] = 3.99e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][60] = 7.91e-7 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][70] = 1.09e-6 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][80] = 2.09e-6 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][90] = 4.04e-6 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);
            I_off_n[2][100] = 4.48e-6 * pow(SUPPLY_VOLTAGE / (vdd[2]), 5);

            I_g_on_n[2][0] = 2.74e-9;  // A/micron
            I_g_on_n[2][10] = 2.74e-9;
            I_g_on_n[2][20] = 2.74e-9;
            I_g_on_n[2][30] = 2.74e-9;
            I_g_on_n[2][40] = 2.74e-9;
            I_g_on_n[2][50] = 2.74e-9;
            I_g_on_n[2][60] = 2.74e-9;
            I_g_on_n[2][70] = 2.74e-9;
            I_g_on_n[2][80] = 2.74e-9;
            I_g_on_n[2][90] = 2.74e-9;
            I_g_on_n[2][100] = 2.74e-9;

            if (TECH_TYPE == comm_dram) {
                I_off_n[3][0] = 1.1e-13;  // A/micron
                I_off_n[3][10] = 2.11e-13;
                I_off_n[3][20] = 3.88e-13;
                I_off_n[3][30] = 6.9e-13;
                I_off_n[3][40] = 1.19e-12;
                I_off_n[3][50] = 1.98e-12;
                I_off_n[3][60] = 3.22e-12;
                I_off_n[3][70] = 5.09e-12;
                I_off_n[3][80] = 7.85e-12;
                I_off_n[3][90] = 1.18e-11;
                I_off_n[3][100] = 1.72e-11;
            } else if (TECH_TYPE == lp_dram) {
                std::cout << "22nm LP DRAM unavailable in McPAT" << std::endl;
                exit(0);
            }
        }

        if (tech == 16) {
            // Empirical undifferetiated core/FU coefficient
            curr_logic_scaling_co_eff = 0.7 * 0.7 * 0.7 * 0.7 * 0.7;
            curr_core_tx_density = 1.25 / 0.7 / 0.7 / 0.7;

            vdd[0] = 0.7;
            n_to_p_eff_curr_drv_ratio[0] =
                2;  // Wpmos/Wnmos = 2 in 2007 MASTAR. Look in
            I_off_n[0][0] = 1.52e-7 / 1.5 * 1.2 * 1.07;
            I_off_n[0][10] = 1.55e-7 / 1.5 * 1.2 * 1.07;
            I_off_n[0][20] = 1.59e-7 / 1.5 * 1.2 * 1.07;
            I_off_n[0][30] = 1.68e-7 / 1.5 * 1.2 * 1.07;
            I_off_n[0][40] = 1.90e-7 / 1.5 * 1.2 * 1.07;
            I_off_n[0][50] = 2.69e-7 / 1.5 * 1.2 * 1.07;
            I_off_n[0][60] = 5.32e-7 / 1.5 * 1.2 * 1.07;
            I_off_n[0][70] = 1.02e-6 / 1.5 * 1.2 * 1.07;
            I_off_n[0][80] = 1.62e-6 / 1.5 * 1.2 * 1.07;
            I_off_n[0][90] = 2.73e-6 / 1.5 * 1.2 * 1.07;
            I_off_n[0][100] = 6.1e-6 / 1.5 * 1.2 * 1.07;
            // for 16nm DG HP
            I_g_on_n[0][0] = 1.07e-9;  // A/micron
            I_g_on_n[0][10] = 1.07e-9;
            I_g_on_n[0][20] = 1.07e-9;
            I_g_on_n[0][30] = 1.07e-9;
            I_g_on_n[0][40] = 1.07e-9;
            I_g_on_n[0][50] = 1.07e-9;
            I_g_on_n[0][60] = 1.07e-9;
            I_g_on_n[0][70] = 1.07e-9;
            I_g_on_n[0][80] = 1.07e-9;
            I_g_on_n[0][90] = 1.07e-9;
            I_g_on_n[0][100] = 1.07e-9;

            //    	vdd[1] = 0.8;
            //    	n_to_p_eff_curr_drv_ratio[1] = 2;
            //    	I_off_n[1][0] = 2.43e-11;
            //    	I_off_n[1][10] = 4.85e-11;
            //    	I_off_n[1][20] = 9.68e-11;
            //    	I_off_n[1][30] = 1.94e-10;
            //    	I_off_n[1][40] = 3.87e-10;
            //    	I_off_n[1][50] = 7.73e-10;
            //    	I_off_n[1][60] = 3.55e-10;
            //    	I_off_n[1][70] = 3.09e-9;
            //    	I_off_n[1][80] = 6.19e-9;
            //    	I_off_n[1][90] = 1.24e-8;
            //    	I_off_n[1][100]= 2.48e-8;
            //
            //    	//    for 22nm LSTP HP
            //    	I_g_on_n[1][0]  = 4.51e-10;//A/micron
            //    	I_g_on_n[1][10] = 4.51e-10;
            //    	I_g_on_n[1][20] = 4.51e-10;
            //    	I_g_on_n[1][30] = 4.51e-10;
            //    	I_g_on_n[1][40] = 4.51e-10;
            //    	I_g_on_n[1][50] = 4.51e-10;
            //    	I_g_on_n[1][60] = 4.51e-10;
            //    	I_g_on_n[1][70] = 4.51e-10;
            //    	I_g_on_n[1][80] = 4.51e-10;
            //    	I_g_on_n[1][90] = 4.51e-10;
            //    	I_g_on_n[1][100] = 4.51e-10;

            if (TECH_TYPE == comm_dram) {
                I_off_n[3][0] = 1.1e-13;  // A/micron
                I_off_n[3][10] = 2.11e-13;
                I_off_n[3][20] = 3.88e-13;
                I_off_n[3][30] = 6.9e-13;
                I_off_n[3][40] = 1.19e-12;
                I_off_n[3][50] = 1.98e-12;
                I_off_n[3][60] = 3.22e-12;
                I_off_n[3][70] = 5.09e-12;
                I_off_n[3][80] = 7.85e-12;
                I_off_n[3][90] = 1.18e-11;
                I_off_n[3][100] = 1.72e-11;
            } else if (TECH_TYPE == lp_dram || TECH_TYPE == itrs_lop ||
                       TECH_TYPE == itrs_lstp) {
                std::cout << "16nm LP DRAM or ITRS LOP or ITRS LSTP "
                             "unavailable in McPAT"
                          << std::endl;
                exit(0);
            }
        }

        // Empirical undifferetiated core/FU coefficient
        g_tp.scaling_factor.logic_scaling_co_eff +=
            curr_alpha * curr_logic_scaling_co_eff;
        g_tp.scaling_factor.core_tx_density +=
            curr_alpha * curr_core_tx_density;
        g_tp.min_w_nmos_ = (3 * technology) / 2;
        g_tp.peri_global.n_to_p_eff_curr_drv_ratio +=
            curr_alpha * n_to_p_eff_curr_drv_ratio[TECH_TYPE];
        g_tp.peri_global.I_off_n +=
            curr_alpha *
            I_off_n
                [TECH_TYPE]
                [TEMPERATURE -
                 300];  //*pow(g_tp.peri_global.Vdd/g_tp.peri_global.Vdd_default,3);//Consider
                        // the voltage change may affect the current density
                        // as well. TODO: polynomial curve-fitting based on
                        // MASTAR may not be accurate enough
        g_tp.peri_global.I_off_p +=
            curr_alpha *
            I_off_n
                [TECH_TYPE]
                [TEMPERATURE -
                 300];  //*pow(g_tp.peri_global.Vdd/g_tp.peri_global.Vdd_default,3);//To
                        // mimic the Vdd effect on Ioff (for the same device,
                        // dvs should not change default Ioff---only changes
                        // if device is different?? but MASTAR shows different
                        // results)
        g_tp.peri_global.I_g_on_n +=
            curr_alpha * I_g_on_n[TECH_TYPE][TEMPERATURE - 300];
        g_tp.peri_global.I_g_on_p +=
            curr_alpha * I_g_on_n[TECH_TYPE][TEMPERATURE - 300];
    }
    return g_tp;
}

void replaceAllInstancesOf(std::string& str, const char* from, const std::string& to) {
    size_t pos = str.find(from);
    while (pos != std::string::npos) {
        str.replace(pos, strlen(from), to);
        pos = str.find(from, pos + to.length());
    }
}

CACTIResult getCACTIResult(std::string filename, size_t vliwWidth) {
    // Read CACTI input file "filename"
    std::string filenameWithDirectory = "cacti/" + filename;
    std::ifstream file(filenameWithDirectory);
    std::string fileContent;
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        fileContent = buffer.str();
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filenameWithDirectory << std::endl;
        exit(1);
    }

    // Replace all instances of ${VLIW} in the file with the actual vliwWidth
    replaceAllInstancesOf(fileContent, "${VLIW}", std::to_string(vliwWidth));
    // Replace all instances of ${BUS_WIDTH} with vliwWidth * 5
    replaceAllInstancesOf(fileContent, "${BUS_WIDTH}", std::to_string(vliwWidth * 5));
    replaceAllInstancesOf(fileContent, "${TEMPERATURE}", std::to_string(TEMPERATURE));
    replaceAllInstancesOf(fileContent, "${TECHNOLOGY}", std::to_string(((double) TARGET_TECHNOLOGY) / MICRO_TO_NANO_ORDER_OF_MAGNITUDE));

    // Write modified input to a temporary file
    std::string tempFilename = filename + ".tmp";
    std::string tempFilenameWithDirectory = filenameWithDirectory + ".tmp";
    // Create the temporary file if it does not exist
    std::ofstream ofs(tempFilenameWithDirectory, std::ios::app);
    ofs.close();
    std::ofstream tempFile(tempFilenameWithDirectory);
    if (tempFile.is_open()) {
        tempFile << fileContent;
        tempFile.close();
    } else {
        std::cerr << "Unable to open temporary file: " << tempFilenameWithDirectory << std::endl;
        exit(1);
    }

    // Execute CACTI program with the input file and pipe output to output file
    std::string outputFilename = filename + ".output";
    std::string outputFilenameWithDirectory = "cacti/" + outputFilename;
    std::string command = "cd cacti; ./cacti -infile " + tempFilename + " > " + outputFilename + "; cd ..";
    int result_code = system(command.c_str());
    if (result_code != 0) {
        std::cerr << "CACTI execution failed with code: " << result_code << std::endl;
        exit(1);
    }

    // Read CACTI output file "output_file" and parse the results
    std::ifstream output_file(outputFilenameWithDirectory);
    std::string line;
    CACTIResult result;

    if (output_file.is_open()) {
        while (getline(output_file, line)) {
            if (line.find("Access time (ns):") != std::string::npos) {
                // Not used
            } else if (line.find("Cycle time (ns):") != std::string::npos) {
                // Not used
            } else if (line.find("Total dynamic read energy per access (nJ):") != std::string::npos) {
                std::string value = line.substr(line.find(":") + 1);
                result.dynamic_read_energy_per_access = std::stod(value);
            } else if (line.find("Total dynamic write energy per access (nJ):") != std::string::npos) {
                std::string value = line.substr(line.find(":") + 1);
                result.dynamic_write_energy_per_access = std::stod(value);
            } else if (line.find("Total leakage power of a bank (mW):") != std::string::npos) {
                std::string value = line.substr(line.find(":") + 1);
                result.leakage_power = std::stod(value);
            } else if (line.find("Total gate leakage power of a bank (mW):") != std::string::npos) {
                std::string value = line.substr(line.find(":") + 1);
                result.gate_leakage_power = std::stod(value);
            } else if (line.find("Cache height x width (mm):") != std::string::npos) {
                std::string value = line.substr(line.find(":") + 1);
                std::stringstream ss(value);
                std::string height_str, width_str;
                getline(ss, height_str, 'x');
                getline(ss, width_str);
                result.height = std::stod(height_str);
                result.width = std::stod(width_str);
            }
        }
        output_file.close();
    } else {
        std::cerr << "Unable to open file: " << outputFilenameWithDirectory << std::endl;
        exit(1);
    }
    return result;
}

double pmos_to_nmos_sz_ratio(TechnologyParameter g_tp) {
    return g_tp.peri_global.n_to_p_eff_curr_drv_ratio;
}

double scaleAreaBasedOnTechnology(double area, size_t old_tech,
                                  size_t new_tech) {
    double scaling_factor = getLogicScalingFactor(old_tech, new_tech);
    return area * scaling_factor;
}

double getLogicScalingFactor(size_t source_technology,
                             size_t target_technology) {
    // scaling factors are w.r.t to 90nm technology
    double target_scaling_factor = getTechnologyParams(target_technology)
                .scaling_factor.logic_scaling_co_eff;
    double source_scaling_factor = getTechnologyParams(source_technology)
                .scaling_factor.logic_scaling_co_eff;
    return target_scaling_factor / source_scaling_factor;
}

double simplified_nmos_leakage(double nwidth, TechnologyParameter g_tp) {
    return nwidth * g_tp.peri_global.I_off_n;
}

double simplified_pmos_leakage(double pwidth, TechnologyParameter g_tp) {
    return pwidth * g_tp.peri_global.I_off_p;
}

double cmos_Ig_n(double nWidth, TechnologyParameter g_tp) {
    return nWidth * g_tp.peri_global.I_g_on_n;
}

double cmos_Ig_p(double pWidth, TechnologyParameter g_tp) {
    return pWidth * g_tp.peri_global.I_g_on_p;
}

double cmos_Isub_leakage(double nWidth, double pWidth,
                         TechnologyParameter g_tp) {
    double nmos_leak = simplified_nmos_leakage(nWidth, g_tp);
    double pmos_leak = simplified_pmos_leakage(pWidth, g_tp);
    return (nmos_leak + pmos_leak) / 2;
}

double cmos_Ig_leakage(double nWidth, double pWidth, TechnologyParameter g_tp) {
    double nmos_leak = cmos_Ig_n(nWidth, g_tp);
    double pmos_leak = cmos_Ig_p(pWidth, g_tp);
    return (nmos_leak + pmos_leak) / 2;
}
