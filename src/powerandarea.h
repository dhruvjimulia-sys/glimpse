#pragma once
#include <cctype>
#include <cstddef>
#include <cstdint>
#include "isa.h"

// Memory area and power functions
// Access time (ns): 0.435422
// Cycle time (ns):  0.327345
// Total dynamic read energy per access (nJ): 0.00105031
// Total dynamic write energy per access (nJ): 0.00138934
// Total leakage power of a bank (mW): 0.0879501
// Total gate leakage power of a bank (mW): 0.0137374
// Cache height x width (mm): 0.204824 x 0.0508967
struct CACTIResult {
    double dynamic_read_energy_per_access;
    double dynamic_write_energy_per_access;
    double leakage_power;
    double gate_leakage_power;
    double height;
    double width;
};

class TechnologyParameter {
   public:
    class DeviceType {
       public:
        double C_g_ideal;
        double C_fringe;
        double C_overlap;
        double C_junc;  // C_junc_area
        double C_junc_sidewall;
        double l_phy;
        double l_elec;
        double R_nch_on;
        double R_pch_on;
        double Vdd;
        double Vdd_default;
        double Vth;
        double Vcc_min_default;  // allowed min vcc; for memory cell it is the
                                 // lowest vcc for data retention. for logic it
                                 // is the vcc to balance the leakage reduction
                                 // and wakeup latency; This is the value
                                 // constrained by the IC technology and cannot
                                 // by changed by external/user voltage supply
        double
            Vcc_min;  // same meaning as Vcc_min_default, however, this value is
                      // set by user, once it is lower than Vcc_min_default;
                      // circuit (e.g. SRAM cells) cannot retain state.
        double I_on_n;
        double I_on_p;
        double I_off_n;
        double I_off_p;
        double I_g_on_n;
        double I_g_on_p;
        double C_ox;
        double t_ox;
        double n_to_p_eff_curr_drv_ratio;
        double long_channel_leakage_reduction;
        double Mobility_n;

        DeviceType()
            : C_g_ideal(0),
              C_fringe(0),
              C_overlap(0),
              C_junc(0),
              C_junc_sidewall(0),
              l_phy(0),
              l_elec(0),
              R_nch_on(0),
              R_pch_on(0),
              Vdd(0),
              Vdd_default(0),
              Vth(0),
              Vcc_min(0),
              I_on_n(0),
              I_on_p(0),
              I_off_n(0),
              I_off_p(0),
              I_g_on_n(0),
              I_g_on_p(0),
              C_ox(0),
              t_ox(0),
              n_to_p_eff_curr_drv_ratio(0),
              long_channel_leakage_reduction(0),
              Mobility_n(0) {};
        void reset() {
            C_g_ideal = 0;
            C_fringe = 0;
            C_overlap = 0;
            C_junc = 0;
            l_phy = 0;
            l_elec = 0;
            R_nch_on = 0;
            R_pch_on = 0;
            Vdd = 0;
            Vdd_default = 0;
            Vth = 0;
            Vcc_min_default = 0;
            Vcc_min = 0;
            I_on_n = 0;
            I_on_p = 0;
            I_off_n = 0;
            I_off_p = 0;
            I_g_on_n = 0;
            I_g_on_p = 0;
            C_ox = 0;
            t_ox = 0;
            n_to_p_eff_curr_drv_ratio = 0;
            long_channel_leakage_reduction = 0;
            Mobility_n = 0;
        }

        void display(uint32_t indent = 0);
    };
    class InterconnectType {
       public:
        double pitch;
        double R_per_um;
        double C_per_um;
        double horiz_dielectric_constant;
        double vert_dielectric_constant;
        double aspect_ratio;
        double miller_value;
        double ild_thickness;

        InterconnectType() : pitch(0), R_per_um(0), C_per_um(0) {};

        void reset() {
            pitch = 0;
            R_per_um = 0;
            C_per_um = 0;
            horiz_dielectric_constant = 0;
            vert_dielectric_constant = 0;
            aspect_ratio = 0;
            miller_value = 0;
            ild_thickness = 0;
        }

        void display(uint32_t indent = 0);
    };
    class MemoryType {
       public:
        double b_w;
        double b_h;
        double cell_a_w;
        double cell_pmos_w;
        double cell_nmos_w;
        double Vbitpre;
        double Vbitfloating;  // voltage when floating bitline is supported

        void reset() {
            b_w = 0;
            b_h = 0;
            cell_a_w = 0;
            cell_pmos_w = 0;
            cell_nmos_w = 0;
            Vbitpre = 0;
            Vbitfloating = 0;
        }

        void display(uint32_t indent = 0);
    };

    class ScalingFactor {
       public:
        double logic_scaling_co_eff;
        double core_tx_density;
        double long_channel_leakage_reduction;

        ScalingFactor()
            : logic_scaling_co_eff(0),
              core_tx_density(0),
              long_channel_leakage_reduction(0) {};

        void reset() {
            logic_scaling_co_eff = 0;
            core_tx_density = 0;
            long_channel_leakage_reduction = 0;
        }

        void display(uint32_t indent = 0);
    };

    double ram_wl_stitching_overhead_;
    double min_w_nmos_;
    double max_w_nmos_;
    double max_w_nmos_dec;
    double unit_len_wire_del;
    double FO4;
    double kinv;
    double vpp;
    double w_sense_en;
    double w_sense_n;
    double w_sense_p;
    double sense_delay;
    double sense_dy_power;
    double w_iso;
    double w_poly_contact;
    double spacing_poly_to_poly;
    double spacing_poly_to_contact;

    double w_comp_inv_p1;
    double w_comp_inv_p2;
    double w_comp_inv_p3;
    double w_comp_inv_n1;
    double w_comp_inv_n2;
    double w_comp_inv_n3;
    double w_eval_inv_p;
    double w_eval_inv_n;
    double w_comp_n;
    double w_comp_p;

    double dram_cell_I_on;
    double dram_cell_Vdd;
    double dram_cell_I_off_worst_case_len_temp;
    double dram_cell_C;
    double gm_sense_amp_latch;

    double w_nmos_b_mux;
    double w_nmos_sa_mux;
    double w_pmos_bl_precharge;
    double w_pmos_bl_eq;
    double MIN_GAP_BET_P_AND_N_DIFFS;
    double MIN_GAP_BET_SAME_TYPE_DIFFS;
    double HPOWERRAIL;
    double cell_h_def;

    double chip_layout_overhead;
    double macro_layout_overhead;
    double sckt_co_eff;

    double fringe_cap;

    uint64_t h_dec;

    DeviceType sram_cell;    // SRAM cell transistor
    DeviceType dram_acc;     // DRAM access transistor
    DeviceType dram_wl;      // DRAM wordline transistor
    DeviceType peri_global;  // peripheral global
    DeviceType cam_cell;     // SRAM cell transistor

    DeviceType sleep_tx;  // Sleep transistor cell transistor

    InterconnectType wire_local;
    InterconnectType wire_inside_mat;
    InterconnectType wire_outside_mat;

    ScalingFactor scaling_factor;

    MemoryType sram;
    MemoryType dram;
    MemoryType cam;

    void display(uint32_t indent = 0);

    void reset() {
        dram_cell_Vdd = 0;
        dram_cell_I_on = 0;
        dram_cell_C = 0;
        vpp = 0;

        sense_delay = 0;
        sense_dy_power = 0;
        fringe_cap = 0;
        //    horiz_dielectric_constant = 0;
        //    vert_dielectric_constant  = 0;
        //    aspect_ratio              = 0;
        //    miller_value              = 0;
        //    ild_thickness             = 0;

        dram_cell_I_off_worst_case_len_temp = 0;

        sram_cell.reset();
        dram_acc.reset();
        dram_wl.reset();
        peri_global.reset();
        cam_cell.reset();
        sleep_tx.reset();

        scaling_factor.reset();

        wire_local.reset();
        wire_inside_mat.reset();
        wire_outside_mat.reset();

        sram.reset();
        dram.reset();
        cam.reset();

        chip_layout_overhead = 0;
        macro_layout_overhead = 0;
        sckt_co_eff = 0;
    }
};

double getComputeArea(size_t vliwWidth, bool overhead = true);

double getComputeSubthresholdLeakage(size_t vliwWidth);

double getComputeGateLeakage(size_t vliwWidth);

double getComputeDynamicPower(Program program);

double getMemoryArea(size_t vliwWidth, bool isPipelining);

double getMemorySubthresholdLeakage(size_t vliwWidth, bool isPipelining);

double getMemoryGateLeakage(size_t vliwWidth, bool isPipelining);

double getMemoryDynamicPower(Program program);