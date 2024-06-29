import {ChangeDetectionStrategy, Component, ContentChild, ElementRef, OnDestroy, OnInit, ViewChild, signal} from '@angular/core';
import { RouterLink } from '@angular/router';
import {MatSelectModule} from '@angular/material/select';
import {MatInputModule} from '@angular/material/input';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatDividerModule} from '@angular/material/divider';
import {MatButtonModule} from '@angular/material/button';
import { ChooseBatteryStateComponent } from './stages/choose-battery-state/choose-battery-state.component';
import { ChooseSupervisedComponent } from './stages/choose-supervised/choose-supervised.component';
import { PredictionViewComponent } from './stages/prediction-view/prediction-view.component';
import { EvalutionViewComponent } from './stages/evalution-view/evalution-view.component';
import { ResultViewComponent } from './stages/result-view/result-view.component';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    RouterLink, MatFormFieldModule, MatInputModule, MatSelectModule, MatButtonModule, 
    MatDividerModule, MatIconModule, ChooseBatteryStateComponent, ChooseSupervisedComponent,
    PredictionViewComponent, EvalutionViewComponent, ResultViewComponent
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HomeComponent implements OnInit, OnDestroy{
  // @ViewChild('contentBox') contentBox!: ElementRef<HTMLDivElement>;
  // @ViewChild('matInput') matInput!: ElementRef<HTMLInputElement>;
  catalogueList:string[] = [
    "Chọn bộ dữ liệu học",
    "Chọn danh sách trạng thái của những chu kỳ sạc muốn dự đoán",
    "Tiến hành dự đoán",
    "Kết quả dự đoán",
    "Đánh giá kết quả",
  ];

  currentCatalogueIndex:number = 0;
  disabledPreviousStepBtn:boolean = true;
  disabledResetBtn:boolean = false;
  supervisedBatteryOrders: number[] = [];
  unsupervisedBatteryOrders: number[] = [];
  // predictingStates: number[][] = [];
  predictingBatteryOrder: number = 0;
  // predictingCycleOrder: number = 0;
  result = signal<{
    resultStates:number[][], 
    battery_order:number, 
    mape_Qi:number,
    rmse_Qi:number,
    mape_remain_cycle:number,
    rmse_remain_cycle:number,
  }>({
    // 9 features + cycle_order + Qi prediction + remain_cycle prediction + real cycle + real Qi
    resultStates: [[0.123456789, 1.123456789, 2.123456789, 3.123456789, 4.123456789, 5.123456789, 6.123456789, 7.123456789, 8.123456789 ,9.123456789, 10, 11.123456789, 12.123456789]],  
    battery_order: 1234,
    mape_Qi: 11.123456789,
    rmse_Qi: 12.123456789,
    mape_remain_cycle: 13.123456789,
    rmse_remain_cycle: 14.123456789,
  });

  constructor() {
  }

  ngOnInit() {
  }

  ngOnDestroy() {
  }
  selectCatalogue(index:number){
    if(index < 0 || index >= this.currentCatalogueIndex || this.currentCatalogueIndex === 2 || this.currentCatalogueIndex === 3) return;
    
    if (this.currentCatalogueIndex === 4){
      if (index < 3){
        return;
      }
    }
    
    this.disabledPreviousStepBtn = index === 0 || index === 2 || index === 3;
    this.currentCatalogueIndex = index;
    this.disabledResetBtn = index === 2;

    if(index === 0){
      this.predictingBatteryOrder = 0;
    }
  }
  reset(){
    window.location.reload();
  }
  step1({supervisedBatteryOrders, unsupervisedBatteryOrders}:{supervisedBatteryOrders: number[], unsupervisedBatteryOrders: number[]}){
    this.supervisedBatteryOrders = supervisedBatteryOrders;
    this.unsupervisedBatteryOrders =  unsupervisedBatteryOrders;
    this.currentCatalogueIndex = 1;
    this.disabledPreviousStepBtn = false;
    this.disabledResetBtn = false;

    console.log(this.supervisedBatteryOrders);
  }
  step2({battery_order}:{battery_order:number}){
    // this.predictingState = predictingState;
    this.predictingBatteryOrder = battery_order;
    // this.predictingCycleOrder = cycle_order;

    this.currentCatalogueIndex = 2;
    this.disabledPreviousStepBtn = true;
    this.disabledResetBtn = true;
  }
  step3(result : {
    resultStates:number[][], 
    battery_order:number, 
    mape_Qi:number,
    rmse_Qi:number,
    mape_remain_cycle:number,
    rmse_remain_cycle:number,
  }){
    this.result.set(result);
    this.currentCatalogueIndex = 3;
    this.disabledPreviousStepBtn = true;
    this.disabledResetBtn = false;
  }
  step4(){
    
    this.currentCatalogueIndex = 4;
    this.disabledPreviousStepBtn = false;
    this.disabledResetBtn = false;
  }

}
