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

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    RouterLink, MatFormFieldModule, MatInputModule, MatSelectModule, MatButtonModule, 
    MatDividerModule, MatIconModule, ChooseBatteryStateComponent, ChooseSupervisedComponent,
    PredictionViewComponent, EvalutionViewComponent
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HomeComponent implements OnInit, OnDestroy{
  // @ViewChild('contentBox') contentBox!: ElementRef<HTMLDivElement>;
  // @ViewChild('matInput') matInput!: ElementRef<HTMLInputElement>;
  catalogueList:string[] = [
    "Chọn tập giám sát",
    "Chọn trạng thái muốn dự đoán",
    "Tiến hành dự đoán",
    "Kết quả/Đánh giá dự đoán",
  ];

  currentCatalogueIndex:number = 2;
  disabledPreviousStepBtn:boolean = true;
  disabledResetBtn:boolean = false;
  supervisedBatteryOrders: number[] = [];
  unsupervisedBatteryOrders: number[] = [];
  predictingState: number[] = [];
  predictingBatteryOrder: number = 0;
  predictingCycleOrder: number = 0;
  result = signal<{
    predictingState:number[], 
    battery_order:number, 
    cycle_order:number, Qi:number, 
    remain_cycle:number
  }>({
    predictingState: [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9],
    battery_order: 20,
    cycle_order: 0.1,
    Qi: 0.3,
    remain_cycle: 100
  });

  constructor() {
  }

  ngOnInit() {
  }

  ngOnDestroy() {
  }
  selectCatalogue(index:number){
    if(index < 0 || index >= this.currentCatalogueIndex || this.currentCatalogueIndex === 2 || this.currentCatalogueIndex === 3) return;
    this.disabledPreviousStepBtn = index === 0 || index === 2 || index === 3;
    this.currentCatalogueIndex = index;
    this.disabledResetBtn = index === 2 || index === 3;

    if(index === 0){
      this.predictingState = [];
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
  step2({predictingState, battery_order, cycle_order}:{predictingState:number[], battery_order:number, cycle_order:number}){
    this.predictingState = predictingState;
    this.predictingBatteryOrder = battery_order;
    this.predictingCycleOrder = cycle_order;

    this.currentCatalogueIndex = 2;
    this.disabledPreviousStepBtn = true;
    this.disabledResetBtn = true;
  }
  step3(result :
    {predictingState:number[], battery_order:number, cycle_order:number, Qi:number, remain_cycle:number}){
    this.result.set(result);
    this.currentCatalogueIndex = 3;
    this.disabledPreviousStepBtn = true;
    this.disabledResetBtn = false;
  }

}
