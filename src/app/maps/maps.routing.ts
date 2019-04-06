import { NgModule } from '@angular/core';
import { Routes } from '@angular/router';

//Maps Components
import { VectorMapComponent } from './vector-map/vector-map.component';
import { GoogleMapComponent } from './google-map/google-map.component';

export const MapsRoutes: Routes = [
    {
        path: '',
        children: [
            {
                path: 'vector-map',
                component: VectorMapComponent,
                data: {
                    title: 'Vector Map'
                }
            },
            {
                path: 'google-map',
                component: GoogleMapComponent,
                data: {
                    title: 'Google Map'
                }
            }
        ]
    }
];

