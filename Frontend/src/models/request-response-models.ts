export interface PredictBatteryLifeRequest {
    supervisedBatteryOrders: number[];
    predictingState: number[];
    predictingBatteryOrder: number;
    predictingCycleOrder: number;
}

export interface PredictBatteryLifeResponse {
    isSuccessful: boolean;
    type: "BadRequest"|"PushToQueue"|"PredictingQi"|"PredictingRemainCycle"|"ResultAndEvalution";
    message: string;
    value: number[] | null;
}