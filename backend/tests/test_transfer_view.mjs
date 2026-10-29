import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

// Exercise the shipped renderer, including its state merge between chat turns.
const template = fs.readFileSync(new URL('../app/templates/index.html', import.meta.url), 'utf8');
const renderers = template.slice(template.indexOf('    function renderLogistics(logistics) {'),
  template.indexOf('    async function requestRegularChat('));

function createRenderer() {
  const context = vm.createContext({
    lastTripData: {}, lastTotalCost: null, workspace: {innerHTML: ''}, initialWorkspaceMarkup: '',
    escapeHtml: String, money: value => `${value} ₽`, icon: () => '', destinationArt: () => '',
    renderConditions: () => '', renderWarning: (title, detail) => [title, detail].filter(Boolean).join(': '), durationLabel: String,
    renderHotel: hotel => `hotel-card:${hotel.name}`, formatMarkdown: String,
    renderTripExtras: () => '', renderDiagnosticReasoning: () => {}, showRefinementChips: () => {},
  });
  vm.runInContext(renderers, context);
  return context;
}

test('transfer edits update prices and logistics while retaining destination and hotel cards', () => {
  const context = createRenderer();
  const paid = {
    destination: {id: 'sochi', name: 'Сочи'}, compared_hotels: [{name: 'Medovaya Hotel'}],
    cart: {hotel: 'Medovaya Hotel', destination_id: 'sochi', destination: 'Сочи', dates: '12–17 ноября'},
    budget_breakdown: {mandatory: {total_mandatory_rub: 100000}, total_trip_budget_range_rub: {min: 120000, max: 140000}},
    logistics: {compatible: true, transfer: {type: 'car', price_rub: 2000}, timeline: {hotel_arrival: '05:18'}},
  };
  context.renderTrip(paid);
  assert.match(context.workspace.innerHTML, /100000 ₽/);
  const removed = {
    cart: {...paid.cart, without_transfer: true}, trip_conditions: {without_transfer: true},
    budget_breakdown: {mandatory: {total_mandatory_rub: 98000}, total_trip_budget_range_rub: {min: 118000, max: 138000}},
    logistics: {compatible: true, transfer: {type: 'self', price_rub: 0}, timeline: {hotel_arrival: null}},
  };
  context.renderTrip(removed);
  const html = context.workspace.innerHTML;
  assert.match(html, /98000 ₽/);
  assert.match(html, /118000 ₽–138000 ₽/);
  assert.match(html, /hotel-card:Medovaya Hotel/);
  assert.match(html, /Самостоятельный трансфер/);
  assert.doesNotMatch(html, /05:18|проживание и трансфер уже состыкованы/);
  assert.match(html, /Есть нюансы/);
  assert.equal(context.lastTripData.logistics.transfer.type, 'self');
  context.renderTrip({budget_breakdown: paid.budget_breakdown, logistics: paid.logistics,
    cart: {...paid.cart, without_transfer: false}});
  assert.match(context.workspace.innerHTML, /100000 ₽/);
  assert.match(context.workspace.innerHTML, /05:18/);
  assert.doesNotMatch(context.workspace.innerHTML, /Самостоятельный трансфер/);
  assert.match(context.workspace.innerHTML, /hotel-card:Medovaya Hotel/);
});

test('corrected saved route title replaces the cached catalogue region after a partial response', () => {
  const context = createRenderer();
  const cart = {destination_id: 'sochi', destination: 'Сочи и Красная Поляна', hotel: 'Medovaya Hotel'};
  context.renderTrip({destination: {id: 'sochi', name: 'Сочи и Красная Поляна'}, cart,
    compared_hotels: [{name: 'Medovaya Hotel'}]});
  context.renderTrip({cart: {...cart, destination: 'Сочи'}});
  assert.match(context.workspace.innerHTML, /<h2>Сочи<\/h2>/);
  assert.doesNotMatch(context.workspace.innerHTML, /Сочи и Красная Поляна/);
  assert.match(context.workspace.innerHTML, /hotel-card:Medovaya Hotel/);
});

test('critical connection warning is visible without a checked or guaranteed package claim', () => {
  const context = createRenderer();
  context.renderTrip({cart: {destination: 'Грузия', total_mandatory_rub: 100000},
    logistics: {compatible: true, flight: {flight_type: 'connecting'}, timeline: {layover_details: {
      is_connecting: true, city: 'Минеральные Воды', duration_minutes: 75,
      risk_evaluation: 'КРИТИЧЕСКИЙ РИСК: самостоятельная пересадка и повторная сдача багажа',
    }}}});
  assert.match(context.workspace.innerHTML, /КРИТИЧЕСКИЙ РИСК/);
  assert.match(context.workspace.innerHTML, /Есть нюансы/);
  assert.doesNotMatch(context.workspace.innerHTML, /safe-badge|>Проверено|уже состыкованы/);
});
