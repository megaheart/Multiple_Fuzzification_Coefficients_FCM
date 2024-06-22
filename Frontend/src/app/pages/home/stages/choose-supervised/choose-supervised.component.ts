import { AfterViewInit, Component, EventEmitter, OnInit, Output, ViewChild, ElementRef, Input } from '@angular/core';
import {MatPaginator, MatPaginatorModule} from '@angular/material/paginator';
import {MatTableDataSource, MatTableModule} from '@angular/material/table';
import {MatSelectModule} from '@angular/material/select';
import {MatInputModule} from '@angular/material/input';
import {MatFormFieldModule} from '@angular/material/form-field';
import { BatteryInfo, BatteryCycleState } from '../../../../../models/models';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import { SelectionModel } from '@angular/cdk/collections';
import {MatCheckboxModule} from '@angular/material/checkbox';

@Component({
  selector: 'app-choose-supervised',
  standalone: true,
  imports: [
    MatTableModule, MatPaginatorModule, MatFormFieldModule, MatInputModule, MatSelectModule,
    FormsModule, MatButtonModule, MatCheckboxModule
  ],
  templateUrl: './choose-supervised.component.html',
  styleUrl: './choose-supervised.component.scss'
})
export class ChooseSupervisedComponent implements OnInit, AfterViewInit {
  displayedColumns: string[] = ['select', 'battery_order', 'policy_readable', 'cycle_count'];
  dataSource : MatTableDataSource<BatteryInfo> = new MatTableDataSource<BatteryInfo>([]);
  numOfSelectingBatteryOrdersRandom: number = 115;
  selection: SelectionModel<BatteryInfo>;

  @ViewChild(MatPaginator) paginator!: MatPaginator;

  @Input() supervisedBatteryOrders: number[] = [];
  @Output() nextStepEvent = new EventEmitter<{supervisedBatteryOrders: number[], unsupervisedBatteryOrders: number[]}>();

  @ViewChild('rightLayout') rightLayout!: ElementRef<HTMLDivElement>;
  @ViewChild('dataTable') dataTable!: ElementRef<HTMLDivElement>;
  @ViewChild('tableBlock') tableBlock!: ElementRef<HTMLTableElement>;

  constructor(private http: HttpClient) { 
    this.selection = new SelectionModel<BatteryInfo>(true, []);
  }

  ngOnInit() {
    let load = () => {
      this.loadDataSource(error => {
        if (window.confirm("Lỗi khi tải dữ liệu từ server, chọn OK để tải lại trang")) {
          load();
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
    this.http.get<BatteryInfo[]>('/api/BatteryInfos').subscribe(data => {
      this.dataSource = new MatTableDataSource(data);
      this.dataSource.paginator = this.paginator;

      if(this.supervisedBatteryOrders.length > 0){
        this.dataSource.data.forEach(row => {
          if(this.supervisedBatteryOrders.includes(row.battery_order)){
            this.selection.select(row);
          }
        });
      }
    }, error, complete);
  }

  chooseBatteryOrdersRandom(){
    if(this.numOfSelectingBatteryOrdersRandom < 50 || this.numOfSelectingBatteryOrdersRandom > 123) {
      alert("Chọn số lượng pin từ 50 đến 123");
      return;
    }

    let batteryOrders: number[] = Array.from({length: this.dataSource.data.length}, (_, i) => i + 1);
    for(let i = 0; i < this.numOfSelectingBatteryOrdersRandom; i++){
      // randomIndex from i to batteryOrders.length - 1
      let randomIndex = i + Math.floor(Math.random() * (batteryOrders.length - i));
      let temp = batteryOrders[i];
      batteryOrders[i] = batteryOrders[randomIndex];
      batteryOrders[randomIndex] = temp;
    }
    batteryOrders = batteryOrders.slice(0, this.numOfSelectingBatteryOrdersRandom);
    this.selection.clear();
    this.dataSource.data.forEach(row => {
      if(batteryOrders.includes(row.battery_order)){
        this.selection.select(row);
      }
    });
  }

  nextStep(){
    if(this.selection.selected.length < 50 || this.selection.selected.length > 123){
      alert("Chọn số lượng pin từ 50 đến 123");
      return;
    }
    
    let supervisedBatteryOrders: number[] = [];
    let unsupervisedBatteryOrders: number[] = [];
    this.selection.selected.forEach(row => {
      supervisedBatteryOrders.push(row.battery_order);
    });
    

    Array.from({length: this.dataSource.data.length}, (_, i) => i + 1).forEach(i => {
      if(!supervisedBatteryOrders.includes(i)){
        unsupervisedBatteryOrders.push(i);
      }
    });

    this.nextStepEvent.emit({supervisedBatteryOrders, unsupervisedBatteryOrders});
  }

  /** Whether the number of selected elements matches the total number of rows. */
  isAllSelected() {
    const numSelected = this.selection.selected.length;
    const numRows = this.dataSource.data.length;
    return numSelected == numRows;
  }

  /** Selects all rows if they are not all selected; otherwise clear selection. */
  toggleAllRows() {
    this.isAllSelected() ?
        this.selection.clear() :
        this.dataSource.data.forEach(row => this.selection.select(row));
  }
}
