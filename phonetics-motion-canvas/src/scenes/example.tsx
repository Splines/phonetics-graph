import { makeScene2D, Circle, Layout, Txt } from '@motion-canvas/2d';
import { all, createRef } from '@motion-canvas/core';
import {useScene } from '@motion-canvas/core/lib/utils';

export default makeScene2D(function* (view) {
    const textFill = useScene().variables.get('textFill');
    const circle = createRef<Circle>();
    
    view.add(
        <Layout direction={'column'} alignItems="center" layout>
            <Txt fontFamily={'Charis'} fill={textFill}>p</Txt>
            <Txt fontFamily={'Charis'} fill={textFill}>ɥ</Txt>
            <Txt fontFamily={'Charis'} fill={textFill}>i</Txt>
            <Txt fontFamily={'Charis'} fill={textFill}>s</Txt>
            <Txt fontFamily={'Charis'} fill={textFill}>ɑ̃</Txt>
            <Txt fontFamily={'Charis'} fill={textFill}>s</Txt>
            {/* <Txt fontFamily={'Charis'} fill={textFill}>nɥɑ̃s</Txt> */}
        </Layout>
    );
});