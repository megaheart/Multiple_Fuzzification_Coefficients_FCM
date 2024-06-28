import { AfterViewInit, Component, EventEmitter, OnInit, Output, ViewChild, ElementRef, Input, Signal, signal, computed } from '@angular/core';
import { MatPaginator, MatPaginatorModule } from '@angular/material/paginator';
import { MatTableDataSource, MatTableModule } from '@angular/material/table';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { BatteryCycleState} from '../../../../../models/models';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { SelectionModel } from '@angular/cdk/collections';
import { MatCheckboxModule } from '@angular/material/checkbox';
import {MatRadioModule} from '@angular/material/radio';
import { MatTooltipModule } from '@angular/material/tooltip'; 
import { DecimalPipe } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-result-view',
  standalone: true,
  imports: [MatTableModule, MatPaginatorModule, MatFormFieldModule, MatInputModule, MatSelectModule,
    FormsModule, MatButtonModule, MatCheckboxModule, MatRadioModule, MatTooltipModule,
    DecimalPipe],
  templateUrl: './result-view.component.html',
  styleUrl: './result-view.component.scss'
})
export class ResultViewComponent implements AfterViewInit {
  @Input() result : Signal<{
    resultStates:number[][], 
    battery_order:number, 
    mape_Qi:number,
    rmse_Qi:number,
    mape_remain_cycle:number,
    rmse_remain_cycle:number,
  }> = signal({
    // 9 features + cycle_order + Qi prediction + remain_cycle prediction + real cycle + real Qi
    resultStates: [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
    battery_order: -1,
    mape_Qi: -1,
    rmse_Qi: -1,
    mape_remain_cycle: -1,
    rmse_remain_cycle: -1,
  });

  @Output() nextStepEvent = new EventEmitter<void>();

  @ViewChild('rightLayout') rightLayout!: ElementRef<HTMLDivElement>;
  @ViewChild('dataTable') dataTable!: ElementRef<HTMLDivElement>;
  @ViewChild('tableBlock') tableBlock!: ElementRef<HTMLTableElement>;
  @ViewChild(MatPaginator) paginator!: MatPaginator;

  props = PROPS;
  displayedColumns = PROPS.map(r => r.propertyName);
  paginator_signal = signal<MatPaginator | null>(null)
  dataSource = computed(() => {
    let data = this.result().resultStates.map((row, index) => {
      let obj:ResultStateInfo = {
        c1a_I_dt: row[0],
        c1a_avg_T: row[1],
        c1a_avg_I: row[2],
        c1_max_I: row[3],
        c2_max_I: row[4],
        c1_max_T: row[5],
        c1_min_T: row[6],
        c2_max_T: row[7],
        c2_min_T: row[8],
        Qi_prediction: row[10],
        remain_cycle_prediction: row[11],
      };
      return obj;
    });

    let s = new MatTableDataSource(data);
    s.paginator = this.paginator_signal();
    return s;
  });

  ngAfterViewInit() {
    // console.log(this.rightLayout);
    // console.log(this.dataTable);
    var rect = this.rightLayout.nativeElement.getBoundingClientRect();
    this.rightLayout.nativeElement.style.maxHeight = rect.height + 'px';
    this.rightLayout.nativeElement.style.maxWidth = rect.width + 'px';
    this.dataTable.nativeElement.style.overflow = 'hidden';
    this.dataTable.nativeElement.style.display = 'grid';

    this.paginator_signal.set(this.paginator);
  }

  nextStep() {
    this.nextStepEvent.emit();
  }
}

type ResultStateInfo = {
  Qi_prediction: number;
  remain_cycle_prediction: number;
  c1a_I_dt: number;
  c1a_avg_T: number;
  c1a_avg_I: number;
  c1_max_I: number;
  c2_max_I: number;
  c1_max_T: number;
  c1_min_T: number;
  c2_max_T: number;
  c2_min_T: number;
};

const PROPS : {
  columnName: string,
  propertyName: string,
  indexProp: number,
  tooltip: string,
  unit: string,
  isNotDecimal?: boolean,
  isInteger?: boolean,
  minWidth?: number,
}[] = [
  {
    columnName: 'Qi_pred',
    propertyName: 'Qi_prediction',
    indexProp: 10,
    tooltip: 'Dung lượng Pin dự đoán',
    unit: 'C'
  },
  {
    columnName: 'life_pred',
    propertyName: 'remain_cycle_prediction',
    indexProp: 11,
    tooltip: 'Chu kỳ sống dự đoán',
    unit: 'Chu kỳ',
    isInteger: true
  },
  {
    columnName: 'c1a_I_dt',
    propertyName: 'c1a_I_dt',
    indexProp: 0,
    tooltip: 'Điện lượng Pha 1a',
    unit: 'C'
  },
  {
    columnName: 'c1a_avg_T',
    propertyName: 'c1a_avg_T',
    indexProp: 1,
    tooltip: 'Nhiệt độ trung bình Pha 1a',
    unit: '°C'
  },
  {
    columnName: 'c1a_avg_I',
    propertyName: 'c1a_avg_I',
    indexProp: 2,
    tooltip: 'Cường độ dòng điện trung bình Pha 1a',
    unit: 'A'
  },
  {
    columnName: 'c1_max_I',
    propertyName: 'c1_max_I',
    indexProp: 3,
    tooltip: 'Cường độ dòng điện cực đại Pha 1',
    unit: 'A'
  },
  {
    columnName: 'c2_max_I',
    propertyName: 'c2_max_I',
    indexProp: 4,
    tooltip: 'Cường độ dòng điện cực đại Pha 2',
    unit: 'A'
  },
  {
    columnName: 'c1_max_T',
    propertyName: 'c1_max_T',
    indexProp: 5,
    tooltip: 'Nhiệt độ cực đại Pha 1',
    unit: '°C'
  },
  {
    columnName: 'c1_min_T',
    propertyName: 'c1_min_T',
    indexProp: 6,
    tooltip: 'Nhiệt độ cực tiểu Pha 1',
    unit: '°C'
  },
  {
    columnName: 'c2_max_T',
    propertyName: 'c2_max_T',
    indexProp: 7,
    tooltip: 'Nhiệt độ cực đại Pha 2',
    unit: '°C'
  },
  {
    columnName: 'c2_min_T',
    propertyName: 'c2_min_T',
    indexProp: 8,
    tooltip: 'Nhiệt độ cực tiểu Pha 2',
    unit: '°C'
  }
];