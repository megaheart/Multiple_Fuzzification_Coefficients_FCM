import { Component, OnInit, Input } from '@angular/core';
import { MatTooltipModule } from '@angular/material/tooltip'; 
import { DecimalPipe } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-evalution-view',
  standalone: true,
  imports: [MatButtonModule, MatTooltipModule, DecimalPipe, MatIconModule],
  templateUrl: './evalution-view.component.html',
  styleUrl: './evalution-view.component.scss'
})
export class EvalutionViewComponent implements OnInit{
  @Input() result : {
    predictingState:number[], 
    battery_order:number, 
    cycle_order:number, Qi:number, 
    remain_cycle:number
  } = {
    predictingState: [-1, -1, -1, -1, -1],
    battery_order: -1,
    cycle_order: -1,
    Qi: -1,
    remain_cycle: -1
  };

  props = PROPS;
  constructor() {
  }

  ngOnInit() {
  }
}

const PROPS : {
  columnName: string,
  propertyName: string,
  tooltip: string,
}[] = [
  // {
  //   columnName: 'Pin',
  //   propertyName: 'battery_order',
  //   tooltip: 'Số thứ tự của pin',
  // },
  // {
  //   columnName: 'Chu kỳ',
  //   propertyName: 'cycle_order',
  //   tooltip: 'Thứ tự của chu kỳ',
  // },
  {
    columnName: 'c1a_I_dt',
    propertyName: 'c1a_I_dt',
    tooltip: 'Điện lượng Pha 1a',
  },
  {
    columnName: 'c1a_avg_T',
    propertyName: 'c1a_avg_T',
    tooltip: 'Nhiệt độ trung bình Pha 1a',
  },
  {
    columnName: 'c1a_avg_I',
    propertyName: 'c1a_avg_I',
    tooltip: 'Cường độ dòng điện trung bình Pha 1a',
  },
  {
    columnName: 'c1_max_I',
    propertyName: 'c1_max_I',
    tooltip: 'Cường độ dòng điện cực đại Pha 1',
  },
  {
    columnName: 'c2_max_I',
    propertyName: 'c2_max_I',
    tooltip: 'Cường độ dòng điện cực đại Pha 2',
  },
  {
    columnName: 'c1_max_T',
    propertyName: 'c1_max_T',
    tooltip: 'Nhiệt độ cực đại Pha 1',
  },
  {
    columnName: 'c1_min_T',
    propertyName: 'c1_min_T',
    tooltip: 'Nhiệt độ cực tiểu Pha 1',
  },
  {
    columnName: 'c2_max_T',
    propertyName: 'c2_max_T',
    tooltip: 'Nhiệt độ cực đại Pha 2',
  },
  {
    columnName: 'c2_min_T',
    propertyName: 'c2_min_T',
    tooltip: 'Nhiệt độ cực tiểu Pha 2',
  }
];