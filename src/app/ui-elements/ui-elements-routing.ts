import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

//Ui Elements Components
import { ButtonsComponent } from './buttons/buttons.component';
import { CardsComponent } from './cards/cards.component';
import { ModalsComponent } from './modals/modals.component';
import { TabsComponent } from './tabs/tabs.component';
import { ProgressNSliderComponent } from './progress-slider/progress-slider.component';
import { NotificationComponent } from './notification/notification.component';
import { TypographyComponent } from './typography/typography.component';


export const UiElementsRoutes: Routes = [
    {
        path: '',
        data: {
            title: 'ui-elements'
        },
        children: [
            {
                path: 'cards',
                component: CardsComponent,
                data: {
                    title: 'Cards'
                }
            },
            {
                path: 'buttons',
                component: ButtonsComponent,
                data: {
                    title: 'Buttons'
                }
            },
            {
                path: 'modals',
                component: ModalsComponent,
                data: {
                    title: 'Modals'
                }
            },
            {
                path: 'tabs',
                component: TabsComponent,
                data: {
                    title: 'Tabs & Accordion'
                }
            },
            {
                path: 'progress-slider',
                component: ProgressNSliderComponent,
                data: {
                    title: 'Progress & Slider'
                }
            },
            {
                path: 'notification',
                component: NotificationComponent,
                data: {
                    title: 'Notification'
                }
            },
            {
                path: 'typography',
                component: TypographyComponent,
                data: {
                    title: 'Typography'
                }
            },
            {
                path: 'icons',
                data: {
                    title: 'Icons'
                }
            }
        ]
    }
];
