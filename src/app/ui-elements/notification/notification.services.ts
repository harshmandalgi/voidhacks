import { Injectable } from '@angular/core';
import { Subject }    from 'rxjs/Subject';

@Injectable()
export class NotificationCommunicationService {
    // Observable string sources
    private positionSource = new Subject<string>();

    // Observable string streams
    position$ = this.positionSource.asObservable();

    setPosition(position) {
        this.positionSource.next(position);
    }
}
