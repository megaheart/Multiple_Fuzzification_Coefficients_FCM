export interface BatteryCycleState {
    battery_order: number;
    cycle_order: number;
    c1a_I_dt: number;
    c1a_avg_T: number;
    c1a_avg_I: number;
    c1_max_I: number;
    c2_max_I: number;
    c1_max_T: number;
    c1_min_T: number;
    c2_max_T: number;
    c2_min_T: number;
}

export interface BatteryInfo {
    battery_order: number;
    policy_readable: string;
    cycle_count: number;
}
