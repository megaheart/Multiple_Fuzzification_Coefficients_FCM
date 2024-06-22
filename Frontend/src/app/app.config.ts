import { ApplicationConfig, importProvidersFrom } from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { withInterceptorsFromDi, provideHttpClient } from '@angular/common/http';
import {VERSION as MAT_VERSION, MatNativeDateModule} from '@angular/material/core';

export const appConfig: ApplicationConfig = {
  providers: [
    provideHttpClient(withInterceptorsFromDi()),
    provideRouter(routes), 
    provideAnimationsAsync(),
    importProvidersFrom(MatNativeDateModule),
  ],
};
