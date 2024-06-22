import { AfterViewInit, Component, EventEmitter, OnInit, Output, ViewChild, ElementRef, Input } from '@angular/core';
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
  selector: 'app-choose-battery-state',
  standalone: true,
  imports: [
    MatTableModule, MatPaginatorModule, MatFormFieldModule, MatInputModule, MatSelectModule,
    FormsModule, MatButtonModule, MatCheckboxModule, MatRadioModule, MatTooltipModule,
    DecimalPipe
  ],
  templateUrl: './choose-battery-state.component.html',
  styleUrl: './choose-battery-state.component.scss'
})
export class ChooseBatteryStateComponent implements OnInit, AfterViewInit {
  columnInfos = DISPLAYED_COLUMNS;
  displayedColumns = ['select', ...DISPLAYED_COLUMNS.map(column => column.propertyName)];
  dataSource: MatTableDataSource<BatteryCycleState> = new MatTableDataSource<BatteryCycleState>([]);
  numOfSelectingBatteryOrdersRandom: number = 115;
  selection: SelectionModel<BatteryCycleState>;

  @ViewChild(MatPaginator) paginator!: MatPaginator;
  @Output() nextStepEvent = new EventEmitter<{predictingState:number[], battery_order:number, cycle_order:number}>();
  @Input() unsupervisedBatteryOrders: number[] = [];

  @ViewChild('rightLayout') rightLayout!: ElementRef<HTMLDivElement>;
  @ViewChild('dataTable') dataTable!: ElementRef<HTMLDivElement>;
  @ViewChild('tableBlock') tableBlock!: ElementRef<HTMLTableElement>;

  constructor(private http: HttpClient) {
    this.selection = new SelectionModel<BatteryCycleState>(false, []);
  }

  ngOnInit() {
    let load = () => {
      this.loadDataSource(error => {
        if (window.confirm("Lỗi khi tải dữ liệu từ server, chọn OK để tải lại trang")) {
          window.location.reload();
        }
      });
    }
    
    load();
  }

  ngAfterViewInit() {
    // console.log(this.rightLayout);
    // console.log(this.dataTable);
    var rect = this.rightLayout.nativeElement.getBoundingClientRect();
    this.rightLayout.nativeElement.style.maxHeight = rect.height + 'px';
    this.rightLayout.nativeElement.style.maxWidth = rect.width + 'px';
    this.dataTable.nativeElement.style.overflow = 'hidden';
    this.dataTable.nativeElement.style.display = 'grid';
    
  }

  loadDataSource(error?: ((error: any) => void), complete?:()=>void){
    let unsupervisedBatteryOrdersString = this.unsupervisedBatteryOrders.join(',');
    this.http.get<BatteryCycleState[]>('/api/BatteryInfos/states?batteryOrders=' + unsupervisedBatteryOrdersString).subscribe(data => {
      this.dataSource = new MatTableDataSource(data);
      this.dataSource.paginator = this.paginator;
    }, error, complete);
  }

  chooseBatteryOrdersRandom() {
    if (this.numOfSelectingBatteryOrdersRandom < 50 || this.numOfSelectingBatteryOrdersRandom > 123) {
      alert("Chọn số lượng pin từ 50 đến 123");
      return;
    }

    
    let randomIndex = Math.floor(Math.random() * this.dataSource.data.length);
    let row = this.dataSource.data[randomIndex];
    let pageIndex = Math.floor(randomIndex / this.paginator.pageSize);
    if (pageIndex !== this.paginator.pageIndex) {
      this.paginator.pageIndex = pageIndex;
      this.paginator._changePageSize(this.paginator.pageSize);
    }
    // this.selection.clear();
    this.selection.select(row);
  }

  nextStep() {
    let selectedItem = this.selection.selected[0];
    let predictingState: number[] = [
      selectedItem.c1a_I_dt,
      selectedItem.c1a_avg_T,
      selectedItem.c1a_avg_I,
      selectedItem.c1_max_I,
      selectedItem.c2_max_I,
      selectedItem.c1_max_T,
      selectedItem.c1_min_T,
      selectedItem.c2_max_T,
      selectedItem.c2_min_T
    ]

    this.nextStepEvent.emit({predictingState, battery_order: selectedItem.battery_order, cycle_order: selectedItem.cycle_order});
  }

}

const DISPLAYED_COLUMNS : {
  columnName: string,
  propertyName: string,
  tooltip: string,
  aligh?: 'left' | 'right' | 'center' | 'justify' | 'initial' | 'inherit'
  minWidth?: number,
  isNotDecimal?: boolean,
}[] = [
  {
    columnName: 'Pin',
    propertyName: 'battery_order',
    tooltip: 'Số thứ tự của pin',
    isNotDecimal: true,
  },
  {
    columnName: 'Chu kỳ',
    propertyName: 'cycle_order',
    tooltip: 'Thứ tự của chu kỳ',
    minWidth: 75,
    isNotDecimal: true,
  },
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

