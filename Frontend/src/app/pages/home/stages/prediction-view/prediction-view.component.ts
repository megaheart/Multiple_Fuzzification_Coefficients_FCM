import { Component, Input, Output, signal, WritableSignal, EventEmitter } from '@angular/core';
import * as signalR from "@microsoft/signalr";
import { PredictBatteryLifeRequest, PredictBatteryLifeResponse } from "../../../../../models/request-response-models";
import { SignalrEvents, SignalrHubMethods } from "../../../../../models/enums";
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip'; 

@Component({
  selector: 'app-prediction-view',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule, MatTooltipModule],
  templateUrl: './prediction-view.component.html',
  styleUrl: './prediction-view.component.scss'
})
export class PredictionViewComponent {
  private _hubConnection?: signalR.HubConnection;

  @Input() supervisedBatteryOrders: number[] = [];
  @Input() predictingBatteryOrder: number = 0;

  @Output() nextStepEvent = new EventEmitter<{
    resultStates:number[][], 
    battery_order:number, 
    mape_Qi:number,
    rmse_Qi:number,
    mape_remain_cycle:number,
    rmse_remain_cycle:number,
  }>();

  isServerRunning = false;
  isRetryBtnDisabled = signal(true);
  predictionStages: WritableSignal<{
    name: string,
    code: string,
    status: "PENDING" | "PROCESSING" | "DONE" | "ERROR",
    task?: () => Promise<void>,
    retry?: () => Promise<void>
  }[]>;
  serverResponse: PredictBatteryLifeResponse[] = [];

  constructor() {
    this.predictionStages = this.setupPredictionStages();
  }

  ngOnInit() {
    this.processPrediction();
  }

  ngOnDestroy() {
    this.disconnectFromServer();
  }

  disconnectFromServer() {
    this._hubConnection?.off(SignalrEvents.PREDICT_PROGRESS);
    this._hubConnection?.stop();
  }

  async processPrediction() {
    for (let i = 0; i < this.predictionStages().length; i++) {
      let { status, task, retry } = this.predictionStages()[i];
      if (status === "PENDING" && task) {
        this.predictionStages.update(l => {
          let o = [...l];
          o[i].status = "PROCESSING";
          return o;
        })

        try {
          await task();

          this.predictionStages.update(l => {
            let o = [...l];
            o[i].status = "DONE";
            return o;
          });
          this.isRetryBtnDisabled.set(true);

        } catch (error) {
          this.predictionStages.update(l => {
            let o = [...l];
            o[i].status = "ERROR";
            return o;
          });
          alert(error);
          if (retry) this.isRetryBtnDisabled.set(false);
          this.disconnectFromServer();
          return;
        }
      }
    }

    let res = this.serverResponse.find(x => x.type === 'ResultAndEvalution');

    if(res){
      this.nextStepEvent.emit({
        battery_order: this.predictingBatteryOrder,
        resultStates: res?.values ?? [],
        mape_Qi: res?.value?.at(0) ?? -1,
        rmse_Qi: res?.value?.at(1) ?? -1,
        mape_remain_cycle: res?.value?.at(2) ?? -1,
        rmse_remain_cycle: res?.value?.at(3) ?? -1
      });
    }
    else{
      console.error("ResultAndEvalution not found");
    }
  }


  async connectToServer(): Promise<void> {
    this._hubConnection = new signalR.HubConnectionBuilder()
      .withUrl("/hub")
      .build();
    let conn = this._hubConnection;
    try {
      conn.onclose(() => {
        this.isServerRunning = false;
        console.log("Connection closed");
      });
      await conn.start();
      this.isServerRunning = true;
      conn.on(SignalrEvents.PREDICT_PROGRESS, (res: PredictBatteryLifeResponse) => {
        this.serverResponse.push(res);
        console.log("SignalR", res);
      });
      console.log(conn.state);
    }
    catch (err) {
      console.log('Error while starting connection: ', err);
      throw new Error("Hiện tại không thể kết nối đến máy chủ! Để thử lại vui lòng chọn nút 'Thử lại'.");
    }
  }

  async connectToServerRetry(): Promise<void> {
    this.serverResponse = [];
    this.predictionStages.update(l => l.map(x => {
      return { ...x, status: "PENDING" };
    }));
    await this.processPrediction();
  }

  async waitInQueue(): Promise<void> {
    try {
      await this._hubConnection?.invoke(SignalrHubMethods.PREDICT_BATTERY_LIFE, {
        supervisedBatteryOrders: this.supervisedBatteryOrders,
        predictingBatteryOrder: this.predictingBatteryOrder,
      });
      // await this._hubConnection?.invoke(SignalrHubMethods.PREDICT_BATTERY_LIFE, {
      //   supervisedBatteryOrders: [1, 2, 3, 4, 5],
      //   predictingState: [1, 2, 3, 4, 5],
      //   predictingBatteryOrder: 1,
      //   predictingCycleOrder: 1
      // });
    }
    catch (err) {
      console.error(err);
      throw new Error("Hiện tại không thể thực hiện dự đoán! Để thử lại vui lòng chọn nút 'Thử lại'.");
    }

    for (let i = 0; i < 10; i++) {
      if (this.isServerRunning === false) {
        throw new Error("Mất kết nối đến máy chủ! Để thử lại vui lòng chọn nút 'Thử lại'.");
      }
      let errorRes = this.serverResponse.find(x => x.type === 'BadRequest' || x.isSuccessful === false);
      if (errorRes) {
        throw new Error((errorRes.message + ". Để thử lại vui lòng chọn nút 'Thử lại'.").replace('..', '.'));
      }
      let res = this.serverResponse.find(x => x.type === "PushToQueue");
      if (res) {
        break;
      }
      if (i === 9) {
        throw new Error("Yêu cầu không được hàng đợi chấp nhận! Để thử lại vui lòng chọn nút 'Thử lại'.");
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    while (true) {
      if (this.isServerRunning === false) {
        throw new Error("Mất kết nối đến máy chủ! Để thử lại vui lòng chọn nút 'Thử lại'.");
      }
      let res = this.serverResponse.find(x => x.type === 'PredictingQi');
      if (res) {
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  async waitInQueueRetry(): Promise<void> {
    this._hubConnection = new signalR.HubConnectionBuilder()
      .withUrl("/hub")
      .build();
    this.serverResponse = [];
    this.predictionStages.update(l => l.map(x => {
      return { ...x, status: "PENDING" };
    }));
    await this.processPrediction();
  }

  async predictingQi(): Promise<void> {
    while (true) {
      if (this.isServerRunning === false) {
        throw new Error("Mất kết nối đến máy chủ! Để thử lại vui lòng chọn nút 'Thử lại'.");
      }
      let res = this.serverResponse.find(x => x.type === 'PredictingRemainCycle');
      if (res) {
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  async predictingRemainCycle(): Promise<void> {
    while (true) {
      if (this.isServerRunning === false) {
        throw new Error("Mất kết nối đến máy chủ! Để thử lại vui lòng chọn nút 'Thử lại'.");
      }
      let res = this.serverResponse.find(x => x.type === 'ResultAndEvalution');
      if (res) {

        return;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  public sendMessage(): void {
    // console.log(this.matInput);
    // console.log(this._hubConnection.state, this.matInput.nativeElement.value);

    let json: PredictBatteryLifeRequest = {
      supervisedBatteryOrders: [1, 2, 3, 4, 5],
      predictingState: [1, 2, 3, 4, 5],
      predictingBatteryOrder: 1,
      predictingCycleOrder: 1,
    };

    this._hubConnection?.invoke(SignalrHubMethods.PREDICT_BATTERY_LIFE, json)
      .catch(err => {
        console.error(err);
      });
  }

  retry() {
    this.predictionStages().find(x => x.status === "ERROR")?.retry?.();
  }

  setupPredictionStages() {
    const currentClass = this;
    const _predictionStages: {
      name: string,
      code: string,
      status: "PENDING" | "PROCESSING" | "DONE" | "ERROR",
      task?: () => Promise<void>,
      retry?: () => Promise<void>
    }[] = [
        {
          name: "Kết nối đến máy chủ",
          code: "ConnectingToServer",
          status: "PENDING",
          task: currentClass.connectToServer.bind(currentClass),
          retry: currentClass.connectToServerRetry.bind(currentClass)
        },
        {
          name: "Chờ đến lượt dự đoán trong hàng đợi",
          code: "WaitingInQueue",
          status: "PENDING",
          task: currentClass.waitInQueue.bind(currentClass),
          retry: currentClass.waitInQueueRetry.bind(currentClass)
        },
        {
          name: "Dự đoán dung lượng của Pin",
          code: "PredictingQi",
          status: "PENDING",
          task: currentClass.predictingQi.bind(currentClass),
          retry: currentClass.waitInQueueRetry.bind(currentClass)
        },
        {
          name: "Dự đoán chu kỳ sống của Pin",
          code: "PredictingRemainCycle",
          status: "PENDING",
          task: currentClass.predictingRemainCycle.bind(currentClass),
          retry: currentClass.waitInQueueRetry.bind(currentClass)
        }
      ];

    return signal(_predictionStages);
  }

}

