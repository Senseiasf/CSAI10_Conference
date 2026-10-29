import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const template = fs.readFileSync(new URL('../app/templates/index.html', import.meta.url), 'utf8');
const context = vm.createContext({lastPrompts: new Map(), pick: items => items[0]});
vm.runInContext(template.slice(template.indexOf('    function freshPrompt('),
  template.indexOf('    const refinementChips =')), context);
const modes = vm.runInContext('modeConfig', context);
for (const mode of ['full', 'hotels', 'multi']) {
  test(`${mode} chips never insert unchosen destination, duration, budget or party size`, () => {
    const invented = /Сочи|Полян|Тбилиси|Гудаури|Казбег|Анталь|Стамбул|Турци|Егип|Алта|Дуба|Эмират|ОАЭ|Петербург|Невск|Таиланд|Пхукет|Бангкок|Самуи|Мальдив|\d|вдво[её]м|для пары|для двоих|на неделю|на месяц/i;
    for (const item of modes[mode].examples) {
      assert.doesNotMatch(item.label, invented, item.id);
      for (const variant of item.variants) {
        const inserted = context.freshPrompt(item.id, [variant]);
        assert.doesNotMatch(inserted, invented, `${item.id}: ${inserted}`);
      }
    }
  });
}
