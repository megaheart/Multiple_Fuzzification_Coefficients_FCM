import { Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home.component';
import { PageNotFoundComponent } from './pages/page-not-found/page-not-found.component';
import { TestComponent } from './pages/test/test.component';

export const routes: Routes = [
    { path: '', component: HomeComponent },
    { path: 'test-page', component: TestComponent },
    { path: 'pagenotfound', component: PageNotFoundComponent },
    { /* matcher: pageNotFoundMatch,  */ path: '**', redirectTo: 'pagenotfound', pathMatch: 'full' },
];
