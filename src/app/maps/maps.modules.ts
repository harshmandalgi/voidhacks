import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { ThemeConstants } from '../shared/config/theme-constant';
import { MapsRoutes } from './maps.routing';
import { NguiMapModule} from '@ngui/map';

//Tables Component
import { VectorMapComponent } from './vector-map/vector-map.component';
import { GoogleMapComponent } from './google-map/google-map.component';


@NgModule({
    imports: [
        RouterModule.forChild(MapsRoutes),
        NguiMapModule.forRoot({apiUrl: 'https://maps.googleapis.com/maps/api/js?key=AIzaSyA80cWOalTZWvHNwZWK9VEVVa3gYzOCKJE'})
    ],
    declarations: [
        VectorMapComponent,
        GoogleMapComponent
    ],
    providers: [
        ThemeConstants 
    ]
})
export class MapsModule { }