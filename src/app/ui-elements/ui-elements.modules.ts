import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { PerfectScrollbarModule } from 'ngx-perfect-scrollbar';
import { Cards_Directives } from '../shared/directives/cards.directive';
import { NouisliderModule } from 'ng2-nouislider/src/nouislider';
import { ToastyModule } from 'ng2-toasty';

//Ui Elements Component
import { CardsComponent } from './cards/cards.component';
import { ButtonsComponent } from './buttons/buttons.component';
import { ModalsComponent } from './modals/modals.component';
import { TabsComponent } from './tabs/tabs.component';
import { ProgressNSliderComponent } from './progress-slider/progress-slider.component';
import { NotificationComponent } from './notification/notification.component';
import { NotificationCommunicationService } from './notification/notification.services';
import { TypographyComponent } from './typography/typography.component';


// Ui Elements Routing
import { UiElementsRoutes } from './ui-elements-routing';

@NgModule({
    imports: [
        FormsModule,
        RouterModule.forChild(UiElementsRoutes),
        PerfectScrollbarModule,
        NouisliderModule,
        ToastyModule
    ],
    declarations: [
        Cards_Directives,
        CardsComponent,
        ButtonsComponent,
        ModalsComponent,
        TabsComponent,
        ProgressNSliderComponent,
        NotificationComponent,
        TypographyComponent
    ],
    providers: [ NotificationCommunicationService ]
})

export class UiElementsModule { }